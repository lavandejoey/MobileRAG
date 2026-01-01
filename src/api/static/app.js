// src/api/static/app.js
// MobileRAG Web UI script (WS client + history rendering)
// Author: LIU Ziyi
// License: Apache-2.0

(() => {
    // ---------- Markdown + TeX rendering ----------
    function renderMarkdownToHtml(md) {
        if (!md) return "";
        if (!window.marked) return escapeHtml(md);

        // Configure marked once if you want:
        // marked.setOptions({ gfm: true, breaks: true });

        const raw = window.marked.parse(md);
        if (window.DOMPurify) {
            return window.DOMPurify.sanitize(raw, {
                USE_PROFILES: {html: true},
            });
        }
        return raw;
    }

    function escapeHtml(s) {
        return String(s)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function renderMathInElementSafe(rootEl) {
        if (!rootEl) return;
        if (!window.renderMathInElement) return;
        window.renderMathInElement(rootEl, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
                {left: "\\(", right: "\\)", display: false},
                {left: "\\[", right: "\\]", display: true},
            ],
            throwOnError: false,
        });
    }

    // Throttle rendering during streaming
    let renderScheduled = false;

    function scheduleRender(fn) {
        if (renderScheduled) return;
        renderScheduled = true;
        requestAnimationFrame(() => {
            renderScheduled = false;
            fn();
        });
    }

    const el = {
        prompt: document.getElementById("prompt"),
        composer: document.getElementById("composer"),
        chatList: document.getElementById("chatList"),
        chatItems: document.getElementById("chatItems"),
        newChatBtn: document.getElementById("newChatBtn"),
        statusDot: document.getElementById("statusDot"),
        statusText: document.getElementById("statusText"),
        attachBtn: document.getElementById("attachBtn"),
        actionBtn: document.getElementById("actionBtn"),
        actionIcon: document.getElementById("actionIcon"),
        thinkDrawer: document.getElementById("thinkDrawer"),
        thinkDrawerTitle: document.getElementById("thinkDrawerTitle"),
        thinkDrawerBody: document.getElementById("thinkDrawerBody"),
        thinkDrawerClose: document.getElementById("thinkDrawerClose"),
        thinkOverlay: document.getElementById("thinkOverlay"),
    };

    let ws = null;
    let isBusy = false;

    let selectedChatId = localStorage.getItem("mr_selected_chat_id") || "";

    // Per-turn state (streaming)
    let currentAssistant = null; // {wrap, meta, badge, time, bubble, thinkHintEl, thinkText}
    let turnThinkText = "";
    let turnAnswerText = "";
    let turnThinkMs = 0;

    function flushFinalAnswerRender() {
        if (!currentAssistant) return;
        let ans = currentAssistant.bubble.querySelector(".answer-block");
        if (!ans) {
            ans = document.createElement("div");
            ans.className = "answer-block";
            currentAssistant.bubble.appendChild(ans);
        }
        ans.innerHTML = renderMarkdownToHtml(turnAnswerText);
        renderMathInElementSafe(ans);
        scrollToBottom();
    }

    function epochSecToLocale(tsSec) {
        const n = Number(tsSec);
        if (!Number.isFinite(n) || n <= 0) return "";
        return new Date(n * 1000).toLocaleString();
    }

    function apiBase() {
        return `${location.protocol}//${location.host}`;
    }

    function wsUrl() {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        return `${proto}://${location.host}/v1/chat/ws`;
    }

    function setStatus(kind, text) {
        el.statusDot.className = `dot ${kind}`;
        el.statusText.textContent = text;
    }

    function setActionButtonState() {
        // One mutually-exclusive button:
        // - idle: send icon (disabled when input empty)
        // - busy: stop icon (enabled)
        const text = (el.prompt?.value || "").trim();

        if (isBusy) {
            el.actionBtn.disabled = false;
            el.actionBtn.classList.add("stop");
            el.actionBtn.setAttribute("aria-label", "Stop");
            el.actionIcon.textContent = "■";
            return;
        }

        el.actionBtn.classList.remove("stop");
        el.actionBtn.setAttribute("aria-label", "Send");
        el.actionIcon.textContent = "➤";
        el.actionBtn.disabled = text.length === 0;
    }

    function setBusy(v) {
        isBusy = v;
        setActionButtonState();
        setStatus(v ? "dot-think" : "dot-idle", v ? "Thinking..." : "Idle");
    }

    function nowTime() {
        const d = new Date();
        return d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    }

    function scrollToBottom() {
        el.chatList.scrollTop = el.chatList.scrollHeight;
    }

    function closeWs() {
        if (ws) {
            try {
                ws.close();
            } catch (_) {
            }
        }
        ws = null;
    }

    /* =========================
       Thinking Drawer controls
       ========================= */

    function openThinkDrawer(title, text) {
        el.thinkDrawerTitle.textContent = title || "Thinking";
        el.thinkDrawerBody.textContent = text || "";
        el.thinkDrawer.classList.add("open");
        el.thinkOverlay.classList.add("open");
        el.thinkDrawer.setAttribute("aria-hidden", "false");
        el.thinkOverlay.setAttribute("aria-hidden", "false");
    }

    function closeThinkDrawer() {
        el.thinkDrawer.classList.remove("open");
        el.thinkOverlay.classList.remove("open");
        el.thinkDrawer.setAttribute("aria-hidden", "true");
        el.thinkOverlay.setAttribute("aria-hidden", "true");
    }

    el.thinkDrawerClose.addEventListener("click", closeThinkDrawer);
    el.thinkOverlay.addEventListener("click", closeThinkDrawer);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeThinkDrawer();
    });

    /* =========================
       Thinking hint (meta-line)
       ========================= */

    function makeThinkHint() {
        const span = document.createElement("span");
        span.className = "think-hint hidden";
        // content will be set dynamically
        return span;
    }

    function setThinkHintThinking(hintEl) {
        if (!hintEl) return;
        hintEl.classList.remove("hidden");
        hintEl.style.pointerEvents = "none";
        hintEl.style.textDecoration = "none";
        hintEl.innerHTML = `
            <span class="dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </span>
            <span>Thinking</span>
        `;
    }

    function setThinkHintDone(hintEl, thinkMs, onClick) {
        if (!hintEl) return;
        const secs = (thinkMs / 1000).toFixed(1);
        hintEl.classList.remove("hidden");
        hintEl.style.pointerEvents = "auto";
        hintEl.innerHTML = `Thought · ${secs}s`;
        hintEl.onclick = (e) => {
            e.stopPropagation();
            if (typeof onClick === "function") onClick();
        };
    }

    function hideThinkHint(hintEl) {
        if (!hintEl) return;
        hintEl.classList.add("hidden");
        hintEl.onclick = null;
    }

    /* =========================
       Message rendering
       ========================= */

    function addMessage(role, text, badgeText) {
        const wrap = document.createElement("div");
        wrap.className = "msg";

        const meta = document.createElement("div");
        meta.className = "msg-meta";

        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = badgeText || role.toUpperCase();

        // For assistant messages, add a small clickable hint next to badge
        let thinkHintEl = null;
        if (role === "assistant") {
            thinkHintEl = makeThinkHint();
            badge.appendChild(thinkHintEl);
        }

        const time = document.createElement("span");
        time.textContent = nowTime();

        meta.appendChild(badge);
        meta.appendChild(time);

        const bubble = document.createElement("div");
        bubble.className = `bubble ${role}`;
        bubble.textContent = text || "";

        wrap.appendChild(meta);
        wrap.appendChild(bubble);

        el.chatList.appendChild(wrap);
        scrollToBottom();

        return {wrap, meta, badge, time, bubble, thinkHintEl};
    }

    function ensureAssistantMessage() {
        if (!currentAssistant) {
            currentAssistant = addMessage("assistant", "", "ASSISTANT");
        }
        return currentAssistant;
    }

    function resetTurnState() {
        currentAssistant = null;
        turnThinkText = "";
        turnAnswerText = "";
        turnThinkMs = 0;
    }

    /* =========================
       Chat list refresh/select
       ========================= */

    async function refreshChatList() {
        const r = await fetch(`${apiBase()}/v1/chats?limit=200`, {cache: "no-store"});
        if (!r.ok) return;

        const chats = await r.json();
        el.chatItems.innerHTML = "";

        for (const c of chats) {
            const item = document.createElement("div");
            item.className = `chat-item ${c.chat_id === selectedChatId ? "active" : ""}`;

            const title = document.createElement("div");
            title.className = "chat-item-title";
            title.textContent = c.title || "(no title)";

            const meta = document.createElement("div");
            meta.className = "chat-item-meta";
            meta.innerHTML = `<span>${(c.chat_id || "").slice(0, 8)}</span><span>${epochSecToLocale(c.updated_at)}</span>`;

            const actions = document.createElement("div");
            actions.className = "chat-item-actions";

            const del = document.createElement("button");
            del.className = "chat-del";
            del.type = "button";
            del.textContent = "Delete";
            del.addEventListener("click", async (e) => {
                e.stopPropagation();
                await deleteChat(c.chat_id);
            });

            actions.appendChild(del);

            item.appendChild(title);
            item.appendChild(meta);
            item.appendChild(actions);

            item.addEventListener("click", async () => {
                await selectChat(c.chat_id);
            });

            el.chatItems.appendChild(item);
        }
    }

    async function deleteChat(chatId) {
        await fetch(`${apiBase()}/v1/chats/${chatId}`, {method: "DELETE"});

        if (selectedChatId === chatId) {
            selectedChatId = "";
            localStorage.removeItem("mr_selected_chat_id");
            el.chatList.innerHTML = "";
            resetTurnState();
        }
        await refreshChatList();
    }

    function attachThinkHintToAssistantMsg(assistantMsg, thinkText, metaObj) {
        if (!assistantMsg || !assistantMsg.thinkHintEl) return;
        if (!thinkText) {
            hideThinkHint(assistantMsg.thinkHintEl);
            return;
        }

        const ms = (metaObj && metaObj.think_ms) ? Number(metaObj.think_ms) : 0;
        setThinkHintDone(assistantMsg.thinkHintEl, ms, () => {
            openThinkDrawer(ms > 0 ? `Thinking (${(ms / 1000).toFixed(2)}s)` : "Thinking", thinkText);
        });
    }

    async function selectChat(chatId) {
        selectedChatId = chatId;
        localStorage.setItem("mr_selected_chat_id", selectedChatId);

        el.chatList.innerHTML = "";
        resetTurnState();

        const r = await fetch(`${apiBase()}/v1/chats/${chatId}/messages?limit=2000`, {cache: "no-store"});
        if (!r.ok) {
            await refreshChatList();
            return;
        }
        const msgs = await r.json();

        // Rebuild UI from DB:
        let pendingThink = null;
        let pendingMeta = null;

        let lastAssistantMsg = null;
        let lastAssistantThink = null;

        for (const m of msgs) {
            if (m.role === "user") {
                addMessage("user", m.content, "YOU");
                continue;
            }

            if (m.role === "assistant_think") {
                const thinkText = m.content || "";
                pendingThink = thinkText;

                // If DB order is assistant -> assistant_think, attach retroactively
                if (thinkText && lastAssistantMsg && (!lastAssistantThink)) {
                    lastAssistantThink = thinkText;
                    attachThinkHintToAssistantMsg(lastAssistantMsg, thinkText, pendingMeta);
                }
                continue;
            }

            if (m.role === "meta") {
                try {
                    pendingMeta = JSON.parse(m.content || "{}");
                    // If meta arrives after assistant, update label if already attached
                    if (lastAssistantMsg && lastAssistantThink) {
                        attachThinkHintToAssistantMsg(lastAssistantMsg, lastAssistantThink, pendingMeta);
                    }
                } catch (_) {
                    pendingMeta = null;
                }
                continue;
            }

            if (m.role === "assistant") {
                const assistantMsg = addMessage("assistant", "", "ASSISTANT");
                lastAssistantMsg = assistantMsg;
                lastAssistantThink = null;

                // attach think hint if we already have think
                if (pendingThink) {
                    lastAssistantThink = pendingThink;
                    attachThinkHintToAssistantMsg(assistantMsg, pendingThink, pendingMeta);
                }

                // answer content
                const answer = document.createElement("div");
                answer.className = "answer-block";
                // Render markdown + TeX on reload (IMPORTANT)
                const raw = m.content || "";
                answer.innerHTML = renderMarkdownToHtml(raw);
                renderMathInElementSafe(answer);
                assistantMsg.bubble.appendChild(answer);

                pendingThink = null;
                pendingMeta = null;
            }
        }

        await refreshChatList();
    }

    /* =========================
       Textarea autosize
       ========================= */

    function autoResizeTextarea(textarea, maxHeightPx) {
        const clamp = () => {
            textarea.style.height = "auto";
            const next = Math.min(textarea.scrollHeight, maxHeightPx);
            textarea.style.height = `${next}px`;
            textarea.style.overflowY = textarea.scrollHeight > maxHeightPx ? "auto" : "hidden";
        };
        textarea.addEventListener("input", clamp);
        window.addEventListener("resize", clamp);
        clamp();
        return clamp;
    }

    const clampPromptHeight = el.prompt ? autoResizeTextarea(el.prompt, 180) : () => {
    };

    /* =========================
       WS streaming
       ========================= */

    function startWsStream(message) {
        closeWs();
        resetTurnState();
        setBusy(true);

        // optimistic UI user message
        addMessage("user", message, "YOU");
        el.prompt.value = "";
        clampPromptHeight();
        setActionButtonState();

        ws = new WebSocket(wsUrl());

        ws.onopen = () => {
            ws.send(JSON.stringify({
                session_id: "default",
                chat_id: selectedChatId || undefined,
                message,
            }));
        };

        ws.onmessage = (evt) => {
            let obj;
            try {
                obj = JSON.parse(evt.data);
            } catch (_) {
                return;
            }

            const ev = obj.event;

            if (ev === "chat_created") {
                selectedChatId = obj.chat_id || "";
                localStorage.setItem("mr_selected_chat_id", selectedChatId);
                refreshChatList();
                return;
            }

            if (ev === "think_start") {
                const assistantMsg = ensureAssistantMessage();
                setThinkHintThinking(assistantMsg.thinkHintEl);
                // drawer stays CLOSED by default
                return;
            }

            if (ev === "think_token") {
                const t = obj.token || "";
                turnThinkText += t;
                // do not auto-open drawer, do not auto-scroll drawer
                return;
            }

            if (ev === "think_end") {
                turnThinkMs = Number(obj.think_ms || 0);
                const assistantMsg = ensureAssistantMessage();
                setThinkHintDone(assistantMsg.thinkHintEl, turnThinkMs, () => {
                    openThinkDrawer(
                        turnThinkMs > 0 ? `Thinking (${(turnThinkMs / 1000).toFixed(2)}s)` : "Thinking",
                        turnThinkText
                    );
                });
                return;
            }

            if (ev === "answer_token") {
                const t = obj.token || "";
                turnAnswerText += t;

                const assistantMsg = ensureAssistantMessage();
                let ans = assistantMsg.bubble.querySelector(".answer-block");
                if (!ans) {
                    ans = document.createElement("div");
                    ans.className = "answer-block";
                    assistantMsg.bubble.appendChild(ans);
                }
                // Render Markdown + TeX (throttled)
                scheduleRender(() => {
                    ans.innerHTML = renderMarkdownToHtml(turnAnswerText);
                    renderMathInElementSafe(ans);
                    scrollToBottom();
                });
                return;
            }

            if (ev === "done") {
                flushFinalAnswerRender();
                setBusy(false);
                // closeWs();
                refreshChatList();
                return;
            }

            if (ev === "error") {
                setBusy(false);
                closeWs();
                addMessage("system", `Error: ${obj.error || "unknown"}`, "SYSTEM");
            }
        };

        ws.onerror = () => {
            setBusy(false);
            closeWs();
            addMessage("system", "WebSocket error.", "SYSTEM");
        };

        ws.onclose = () => {
            flushFinalAnswerRender();
            setBusy(false);
            // closeWs();
        };
    }

    function trySend() {
        if (isBusy) return;
        const text = (el.prompt.value || "").trim();
        if (!text) return;
        startWsStream(text);
    }

    function onStop() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        try {
            ws.close();
        } catch (_) {
        }
    }

    /* =========================
       Events
       ========================= */

    // Keep form submit for Enter key flow, but route to trySend()
    el.composer.addEventListener("submit", (e) => {
        e.preventDefault();
        trySend();
    });

    el.actionBtn.addEventListener("click", () => {
        if (isBusy) onStop();
        else trySend();
    });

    el.newChatBtn.addEventListener("click", async () => {
        selectedChatId = "";
        localStorage.removeItem("mr_selected_chat_id");
        el.chatList.innerHTML = "";
        resetTurnState();
        await refreshChatList();
        setStatus("dot-idle", "Idle");
    });

    el.prompt.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            el.composer.requestSubmit();
        }
    });

    // Update disabled/enabled state when typing
    el.prompt.addEventListener("input", () => {
        setActionButtonState();
    });

    /* =========================
       Boot
       ========================= */

    (async () => {
        setStatus("dot-idle", "Idle");
        await refreshChatList();
        if (selectedChatId) {
            await selectChat(selectedChatId);
        } else {
            addMessage("system", "Ready. Select a chat or send a message to create one.", "SYSTEM");
        }
        setActionButtonState();
    })();
})();

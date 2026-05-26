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
        scrollToBottomBtn: document.getElementById("scrollToBottomBtn"),
        homeView: document.getElementById("homeView"),
        chatItems: document.getElementById("chatItems"),
        systemToastHost: document.getElementById("systemToastHost"),
        uploadStrip: document.getElementById("uploadStrip"),
        fileInput: document.getElementById("fileInput"),
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
    let turnCitations = {};
    let currentUploads = [];
    let pendingUploadRequests = new Map();
    let isPreparingSend = false;
    let deletingChatIds = new Set();

    function getChatIdFromPath() {
        const path = window.location.pathname || "/";
        if (path === "/" || path === "") return "";
        const trimmed = path.replace(/^\/+|\/+$/g, "");
        if (!trimmed || trimmed.startsWith("v1") || trimmed.startsWith("static")) return "";
        return decodeURIComponent(trimmed);
    }

    function navigateToChat(chatId, replace = false) {
        const nextPath = chatId ? `/${encodeURIComponent(chatId)}` : "/";
        const currentPath = `${window.location.pathname}${window.location.search}${window.location.hash}`;
        if (currentPath === nextPath) return;
        const fn = replace ? "replaceState" : "pushState";
        window.history[fn]({chatId: chatId || ""}, "", nextPath);
    }

    function clearChatSurface() {
        el.chatList.innerHTML = "";
        resetTurnState();
        updateScrollBottomButton();
    }

    function showHomeView() {
        clearChatSurface();
        el.homeView.classList.add("visible");
        el.homeView.setAttribute("aria-hidden", "false");
        updateScrollBottomButton();
    }

    function hideHomeView() {
        el.homeView.classList.remove("visible");
        el.homeView.setAttribute("aria-hidden", "true");
        updateScrollBottomButton();
    }

    function showSystemToast(message, kind = "error", title = "System") {
        if (!el.systemToastHost) return;
        const toast = document.createElement("div");
        toast.className = `system-toast ${kind}`;

        const titleEl = document.createElement("div");
        titleEl.className = "system-toast-title";
        titleEl.textContent = title;

        const bodyEl = document.createElement("div");
        bodyEl.className = "system-toast-body";
        bodyEl.textContent = message || "";

        toast.appendChild(titleEl);
        toast.appendChild(bodyEl);
        el.systemToastHost.appendChild(toast);

        window.setTimeout(() => {
            toast.remove();
        }, 4500);
    }

    function renderUploadStrip() {
        const items = getPendingUploads();
        el.uploadStrip.innerHTML = "";
        if (!items.length) {
            el.uploadStrip.classList.remove("visible");
            setActionButtonState();
            return;
        }
        el.uploadStrip.classList.add("visible");
        for (const item of items) {
            const chip = document.createElement("div");
            chip.className = `upload-chip ${item.uploading ? "uploading" : (item.processed ? "processed" : "pending")}`;
            chip.title = item.rel_path || item.original_name || item.stored_name || "";

            const name = document.createElement("span");
            name.className = "upload-chip-name";
            name.textContent = item.original_name || item.stored_name || "file";

            const state = document.createElement("span");
            state.className = "upload-chip-state";
            state.textContent = item.uploading ? "uploading" : (item.processed ? "ready" : "uploaded");

            const removeBtn = document.createElement("button");
            removeBtn.className = "upload-chip-remove";
            removeBtn.type = "button";
            removeBtn.setAttribute("aria-label", `Remove ${name.textContent}`);
            removeBtn.textContent = "×";
            removeBtn.disabled = !!item.uploading;
            removeBtn.addEventListener("click", async () => {
                try {
                    await deleteUploadedFile(item.upload_id);
                } catch (err) {
                    showSystemToast(`Upload error: ${err.message || "unknown"}`, "error", "Upload Error");
                }
            });

            chip.appendChild(name);
            chip.appendChild(state);
            chip.appendChild(removeBtn);
            el.uploadStrip.appendChild(chip);
        }
        setActionButtonState();
    }

    function getPendingUploads() {
        return (currentUploads || []).filter((item) => !item.attached_msg_id);
    }

    function hasUploadsInFlight() {
        return pendingUploadRequests.size > 0;
    }

    async function refreshUploads(chatId) {
        if (!chatId) {
            currentUploads = [];
            renderUploadStrip();
            return;
        }
        const r = await fetch(`${apiBase()}/v1/chats/${chatId}/uploads`, {cache: "no-store"});
        if (!r.ok) {
            currentUploads = [];
            renderUploadStrip();
            return;
        }
        currentUploads = await r.json();
        renderUploadStrip();
    }

    async function deleteUploadedFile(uploadId) {
        if (!selectedChatId || !uploadId) return;
        const r = await fetch(`${apiBase()}/v1/chats/${selectedChatId}/uploads/${uploadId}`, {
            method: "DELETE",
        });
        if (!r.ok) {
            const data = await r.json().catch(() => ({}));
            throw new Error(data.detail || "failed to delete upload");
        }
        await refreshUploads(selectedChatId);
    }

    async function ensureDraftChat() {
        if (selectedChatId) return selectedChatId;
        const r = await fetch(`${apiBase()}/v1/chats`, {method: "POST"});
        if (!r.ok) throw new Error("failed to create chat");
        const data = await r.json();
        selectedChatId = data.chat_id || "";
        if (!selectedChatId) throw new Error("missing chat id");
        localStorage.setItem("mr_selected_chat_id", selectedChatId);
        navigateToChat(selectedChatId, false);
        hideHomeView();
        await refreshChatList();
        await refreshUploads(selectedChatId);
        return selectedChatId;
    }

    function flushFinalAnswerRender() {
        if (!currentAssistant) return;
        const shouldStickToBottom = isNearBottom();
        let ans = currentAssistant.bubble.querySelector(".answer-block");
        if (!ans) {
            ans = document.createElement("div");
            ans.className = "answer-block";
            currentAssistant.bubble.appendChild(ans);
        }
        ans.innerHTML = renderAnswerHtml(turnAnswerText, turnCitations);
        renderMathInElementSafe(ans);
        if (shouldStickToBottom) scrollToBottom();
        else updateScrollBottomButton();
    }

    function setTurnCitations(citations) {
        turnCitations = {};
        for (const item of citations || []) {
            if (item && item.citation_id) {
                turnCitations[item.citation_id] = item;
            }
        }
    }

    function replaceCitationTokens(html, citations) {
        const citationKeys = Object.keys(citations || {});
        return String(html || "").replace(/\[([A-Z0-9]{2,12})\]/g, (full, id) => {
            let item = citations[id];
            if (!item) {
                const alias = id.match(/^F(\d+)$/);
                if (alias) {
                    const idx = Number(alias[1]) - 1;
                    if (idx >= 0 && idx < citationKeys.length) {
                        item = citations[citationKeys[idx]];
                    }
                }
            }
            if (!item) return full;
            const title = item.name || id;
            const openUrl = item.open_url || "#";
            const label = compactCitationLabel(title);
            const detail = item.source_label ? `${item.path}\n${item.source_label}` : (item.path || title);
            return `<a class="citation-badge" href="${openUrl}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(detail)}">${escapeHtml(label)}</a>`;
        });
    }

    function renderAnswerHtml(md, citations) {
        const rawHtml = renderMarkdownToHtml(md);
        return replaceCitationTokens(rawHtml, citations || {});
    }

    function compactCitationLabel(name) {
        const base = String(name || "").trim();
        if (!base) return "File";
        const dot = base.lastIndexOf(".");
        const stem = dot > 0 ? base.slice(0, dot) : base;
        const ext = dot > 0 ? base.slice(dot) : "";
        const cleanStem = stem.length > 18 ? `${stem.slice(0, 18).trim()}…` : stem;
        return `${cleanStem}${ext}`;
    }

    function renderMessageUploads(uploads) {
        if (!uploads || !uploads.length) return null;
        const wrap = document.createElement("div");
        wrap.className = "message-upload-list";
        for (const item of uploads) {
            const chip = document.createElement("div");
            chip.className = "message-upload-chip";
            chip.title = item.rel_path || item.original_name || item.stored_name || "";
            chip.textContent = item.original_name || item.stored_name || "file";
            wrap.appendChild(chip);
        }
        return wrap;
    }

    function markUploadsAsSent(uploadIds) {
        if (!uploadIds || !uploadIds.length) return;
        const sentIds = new Set(uploadIds.map((id) => String(id)));
        currentUploads = (currentUploads || []).map((item) => {
            if (!sentIds.has(String(item.upload_id))) return item;
            return {
                ...item,
                attached_msg_id: item.attached_msg_id || "sending",
                uploading: false,
            };
        });
        renderUploadStrip();
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

        const hasPendingUploads = getPendingUploads().length > 0;
        el.actionBtn.classList.remove("stop");
        el.actionBtn.setAttribute("aria-label", "Send");
        el.actionIcon.textContent = "➤";
        el.actionBtn.disabled = isPreparingSend || (text.length === 0 && !hasPendingUploads);
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
        updateScrollBottomButton();
    }

    function isNearBottom(threshold = 56) {
        const remaining = el.chatList.scrollHeight - el.chatList.scrollTop - el.chatList.clientHeight;
        return remaining <= threshold;
    }

    function updateScrollBottomButton() {
        if (!el.scrollToBottomBtn) return;
        const shouldShow = !el.homeView.classList.contains("visible")
            && el.chatList.childElementCount > 0
            && !isNearBottom();
        el.scrollToBottomBtn.classList.toggle("visible", shouldShow);
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

    function shouldIgnoreResumeError(errorText) {
        return errorText === "no_active_turn";
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

    function addMessage(role, text, badgeText, uploads = []) {
        const shouldStickToBottom = isNearBottom();
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
        const uploadList = renderMessageUploads(uploads);
        if (uploadList) {
            bubble.appendChild(uploadList);
        }
        if (text) {
            const textBlock = document.createElement("div");
            textBlock.className = "message-text";
            textBlock.textContent = text;
            bubble.appendChild(textBlock);
        }

        wrap.appendChild(meta);
        wrap.appendChild(bubble);

        el.chatList.appendChild(wrap);
        if (shouldStickToBottom) scrollToBottom();
        else updateScrollBottomButton();

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
        turnCitations = {};
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
            const isDeleting = deletingChatIds.has(c.chat_id);
            if (isDeleting) {
                del.disabled = true;
                del.innerHTML = '<span class="spinner spinner-inline spinner-visible" aria-hidden="true"></span><span>Deleting</span>';
            } else {
                del.textContent = "Delete";
            }
            del.addEventListener("click", async (e) => {
                e.stopPropagation();
                if (deletingChatIds.has(c.chat_id)) return;
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
        deletingChatIds.add(chatId);
        await refreshChatList();
        try {
            await fetch(`${apiBase()}/v1/chats/${chatId}`, {method: "DELETE"});

            if (selectedChatId === chatId) {
                selectedChatId = "";
                localStorage.removeItem("mr_selected_chat_id");
                navigateToChat("", false);
                currentUploads = [];
                renderUploadStrip();
                showHomeView();
            }
        } finally {
            deletingChatIds.delete(chatId);
            await refreshChatList();
        }
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

    async function selectChat(chatId, options = {}) {
        const {updateHistory = true, replaceHistory = false} = options;
        closeWs();
        selectedChatId = chatId;
        localStorage.setItem("mr_selected_chat_id", selectedChatId);
        if (updateHistory) navigateToChat(chatId, replaceHistory);

        hideHomeView();
        clearChatSurface();
        await refreshUploads(chatId);

        const r = await fetch(`${apiBase()}/v1/chats/${chatId}/messages?limit=2000`, {cache: "no-store"});
        if (!r.ok) {
            selectedChatId = "";
            localStorage.removeItem("mr_selected_chat_id");
            navigateToChat("", true);
            currentUploads = [];
            renderUploadStrip();
            showHomeView();
            await refreshChatList();
            return;
        }
        const msgs = await r.json();

        const thinkByTurn = new Map();
        const metaByTurn = new Map();

        for (const m of msgs) {
            if (m.role === "user") {
                addMessage("user", m.content, "YOU", m.uploads || []);
                continue;
            }

            if (m.role === "assistant_think") {
                if (m.turn_id) {
                    thinkByTurn.set(m.turn_id, m.content || "");
                }
                continue;
            }

            if (m.role === "meta") {
                try {
                    if (m.turn_id) {
                        metaByTurn.set(m.turn_id, JSON.parse(m.content || "{}"));
                    }
                } catch (_) {
                    if (m.turn_id) {
                        metaByTurn.set(m.turn_id, null);
                    }
                }
                continue;
            }

            if (m.role === "assistant") {
                const assistantMsg = addMessage("assistant", "", "ASSISTANT");
                const turnThink = m.turn_id ? (thinkByTurn.get(m.turn_id) || "") : "";
                const turnMeta = m.turn_id ? metaByTurn.get(m.turn_id) : null;

                if (turnThink) {
                    attachThinkHintToAssistantMsg(assistantMsg, turnThink, turnMeta);
                }

                // answer content
                const answer = document.createElement("div");
                answer.className = "answer-block";
                // Render markdown + TeX on reload (IMPORTANT)
                const raw = m.content || "";
                const citations = (((turnMeta && turnMeta.citations) || [])).reduce((acc, item) => {
                    if (item && item.citation_id) acc[item.citation_id] = item;
                    return acc;
                }, {});
                answer.innerHTML = renderAnswerHtml(raw, citations);
                renderMathInElementSafe(answer);
                assistantMsg.bubble.appendChild(answer);
            }
        }

        await refreshChatList();
        requestAnimationFrame(() => {
            requestAnimationFrame(() => scrollToBottom());
        });
        await maybeResumeTurn(chatId);
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

    function startWsStream(message, pendingUploads = []) {
        closeWs();
        resetTurnState();
        hideHomeView();
        setBusy(true);
        markUploadsAsSent(pendingUploads.map((item) => item.upload_id));

        // optimistic UI user message
        if (message || pendingUploads.length) {
            addMessage("user", message, "YOU", pendingUploads);
        }
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
                navigateToChat(selectedChatId, false);
                refreshChatList();
                return;
            }

            if (ev === "stage") {
                const stage = obj.stage || "";
                if (stage === "preparing" || stage === "parsing" || stage === "retrieval") {
                    const assistantMsg = ensureAssistantMessage();
                    setThinkHintThinking(assistantMsg.thinkHintEl);
                }
                return;
            }

            if (ev === "rag") {
                setTurnCitations(obj.citations || []);
                return;
            }

            if (ev === "uploads_processed") {
                refreshUploads(selectedChatId);
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
                    const shouldStickToBottom = isNearBottom();
                    ans.innerHTML = renderAnswerHtml(turnAnswerText, turnCitations);
                    renderMathInElementSafe(ans);
                    if (shouldStickToBottom) scrollToBottom();
                    else updateScrollBottomButton();
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
                refreshUploads(selectedChatId);
                if (!shouldIgnoreResumeError(obj.error || "")) {
                    showSystemToast(`Error: ${obj.error || "unknown"}`, "error", "Chat Error");
                }
            }
        };

        ws.onerror = () => {
            setBusy(false);
            closeWs();
            refreshUploads(selectedChatId);
            showSystemToast("WebSocket error.", "error", "Connection Error");
        };

        ws.onclose = () => {
            flushFinalAnswerRender();
            setBusy(false);
            if (selectedChatId) {
                refreshUploads(selectedChatId);
            }
            // closeWs();
        };
    }

    async function waitForPendingUploads() {
        if (!hasUploadsInFlight()) return;
        await Promise.allSettled(Array.from(pendingUploadRequests.values()));
        if (selectedChatId) {
            await refreshUploads(selectedChatId);
        }
    }

    async function maybeResumeTurn(chatId) {
        if (!chatId || isBusy) return;
        closeWs();
        resetTurnState();
        ws = new WebSocket(wsUrl());
        ws.onopen = () => {
            ws.send(JSON.stringify({
                session_id: "default",
                chat_id: chatId,
                message: "",
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
            if (ev === "error") {
                if (!shouldIgnoreResumeError(obj.error || "")) {
                    showSystemToast(`Error: ${obj.error || "unknown"}`, "error", "Chat Error");
                }
                closeWs();
                return;
            }
            if (ev === "chat_created") return;
            if (ev === "stage") {
                setBusy(true);
                const stage = obj.stage || "";
                if (stage === "preparing" || stage === "parsing" || stage === "retrieval") {
                    const assistantMsg = ensureAssistantMessage();
                    setThinkHintThinking(assistantMsg.thinkHintEl);
                }
                return;
            }
            if (ev === "rag") {
                setTurnCitations(obj.citations || []);
                return;
            }
            if (ev === "uploads_processed") {
                refreshUploads(chatId);
                return;
            }
            if (ev === "think_start") {
                setBusy(true);
                const assistantMsg = ensureAssistantMessage();
                setThinkHintThinking(assistantMsg.thinkHintEl);
                return;
            }
            if (ev === "think_token") {
                turnThinkText += obj.token || "";
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
                setBusy(true);
                turnAnswerText += obj.token || "";
                const assistantMsg = ensureAssistantMessage();
                let ans = assistantMsg.bubble.querySelector(".answer-block");
                if (!ans) {
                    ans = document.createElement("div");
                    ans.className = "answer-block";
                    assistantMsg.bubble.appendChild(ans);
                }
                scheduleRender(() => {
                    const shouldStickToBottom = isNearBottom();
                    ans.innerHTML = renderAnswerHtml(turnAnswerText, turnCitations);
                    renderMathInElementSafe(ans);
                    if (shouldStickToBottom) scrollToBottom();
                    else updateScrollBottomButton();
                });
                return;
            }
            if (ev === "done") {
                flushFinalAnswerRender();
                setBusy(false);
                refreshChatList();
                closeWs();
            }
        };
        ws.onerror = () => {
            closeWs();
        };
        ws.onclose = () => {
            flushFinalAnswerRender();
            if (!isBusy) setActionButtonState();
        };
    }

    async function trySend() {
        if (isBusy || isPreparingSend) return;
        const text = (el.prompt.value || "").trim();
        const pendingUploads = getPendingUploads();
        if (!text && !pendingUploads.length) return;
        isPreparingSend = true;
        setActionButtonState();
        try {
            if (hasUploadsInFlight()) {
                setStatus("dot-conn", "Finishing uploads...");
                await waitForPendingUploads();
            }
            const readyUploads = getPendingUploads();
            if (!text && !readyUploads.length) {
                setStatus("dot-idle", "Idle");
                return;
            }
            startWsStream(text, readyUploads);
        } finally {
            isPreparingSend = false;
            if (!isBusy) {
                setActionButtonState();
            }
        }
    }

    function onStop() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        try {
            ws.close();
        } catch (_) {
        }
    }

    async function uploadSelectedFile(file) {
        if (!file) return;
        const chatId = await ensureDraftChat();
        const tempId = `temp-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
        currentUploads = currentUploads.concat([{
            upload_id: tempId,
            chat_id: chatId,
            original_name: file.name || "file",
            stored_name: file.name || "file",
            rel_path: "",
            processed: false,
            attached_msg_id: null,
            uploading: true,
        }]);
        renderUploadStrip();
        const fd = new FormData();
        fd.append("file", file);
        setStatus("dot-think", "Uploading...");
        const uploadPromise = (async () => {
            const r = await fetch(`${apiBase()}/v1/chats/${chatId}/uploads`, {
                method: "POST",
                body: fd,
            });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) {
                throw new Error(data.detail || "upload failed");
            }
            await refreshUploads(chatId);
            await refreshChatList();
        })();
        pendingUploadRequests.set(tempId, uploadPromise);
        try {
            await uploadPromise;
            if (!isPreparingSend && !isBusy) {
                setStatus("dot-idle", "Idle");
            }
        } catch (err) {
            currentUploads = currentUploads.filter((item) => item.upload_id !== tempId);
            renderUploadStrip();
            setStatus("dot-err", "Upload failed");
            showSystemToast(`Upload error: ${err.message || "unknown"}`, "error", "Upload Error");
            throw err;
        } finally {
            pendingUploadRequests.delete(tempId);
            if (selectedChatId && !hasUploadsInFlight() && !isBusy && !isPreparingSend) {
                await refreshUploads(selectedChatId);
            }
            if (el.fileInput) el.fileInput.value = "";
            setActionButtonState();
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

    el.attachBtn.addEventListener("click", () => {
        if (isBusy) return;
        el.fileInput.click();
    });

    el.fileInput.addEventListener("change", async (e) => {
        const file = e.target.files && e.target.files[0];
        try {
            await uploadSelectedFile(file);
        } catch (_) {
        }
    });

    el.chatList.addEventListener("scroll", updateScrollBottomButton);
    el.scrollToBottomBtn.addEventListener("click", () => {
        scrollToBottom();
    });

    el.newChatBtn.addEventListener("click", async () => {
        closeWs();
        selectedChatId = "";
        localStorage.removeItem("mr_selected_chat_id");
        navigateToChat("", false);
        currentUploads = [];
        renderUploadStrip();
        showHomeView();
        await refreshChatList();
        setStatus("dot-idle", "Idle");
    });

    window.addEventListener("popstate", async () => {
        const chatId = getChatIdFromPath();
        closeWs();
        selectedChatId = chatId;
        if (chatId) {
            localStorage.setItem("mr_selected_chat_id", chatId);
            await selectChat(chatId, {updateHistory: false});
        } else {
            localStorage.removeItem("mr_selected_chat_id");
            currentUploads = [];
            renderUploadStrip();
            await refreshChatList();
            showHomeView();
        }
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
        selectedChatId = getChatIdFromPath() || selectedChatId;
        await refreshChatList();
        if (selectedChatId) {
            await selectChat(selectedChatId, {updateHistory: false});
        } else {
            currentUploads = [];
            renderUploadStrip();
            showHomeView();
        }
        setActionButtonState();
    })();
})();

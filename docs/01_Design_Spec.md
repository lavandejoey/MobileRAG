# 一、总体目标与边界

* **形态**：本地离线优先的 Agentic RAG 对话应用（Chatbot），支持 CPU / GPU（6 GB 显存为例）自动切换与手动指定。
* **语种**：中英 1:1，允许跨语检索与跨语回答。
* **多模态**：文本 + 图片。
* **核心数据来源**：文件与图片（本地），对话历史只做“可存可调”而非主要知识源。
* **输出硬约束**：每次回答**必须**带“证据链接”（本地文件路径/页码、图片路径等）。
* **持久化**：向量库本地持久化；离线**增量更新**（新增/修改/删除检测）可稳定运行。
* **复杂度控制**：LangGraph 负责编排与状态持久化（SQLite checkpointer）；LangChain
  完成模型与向量库适配。([LangChain AI][1], [LangChain Blog][2])

---

# 二、模型与向量库选型

## 生成（LLM）

* **Qwen3-1.7B-Base**，上下文长度 **32 K**（本地量化可运行）。([Hugging Face][3], [Qwen][4])

## 文本嵌入（Dense）

* **Qwen3-Embedding-0.6B**，多语、**Instruction-aware**，支持 **MRL（可变维度）**；我们统一采用 1024 维（便于库建模）。可用
  TEI（Text Embeddings Inference）CPU/GPU 服务化或本地 Transformers 推理。([Hugging Face][5], [GitHub][6], [arXiv][7])

## 文本重排（Cross-Encoder）

* **Qwen3-Reranker-0.6B（sequence-classification 变体）**，TopK≈80→TopR≈8–12；该序列分类改造版用于高效打分，避免生成式 logits
  慢的问题。([Hugging Face][8])

## 图像向量（Dense，跨模态）

* **OpenCLIP ViT-B/32（LAION2B）**：文本与图像共享嵌入空间，支持 text↔image 检索。([GitHub][9], [Hugging Face][10])

## 图片自动 Caption

* **BLIP image-captioning-base**（Transformers 原生支持；也有轻量量化变体可选）。Caption
  默认英文，入库做稀疏/稠密融合。([Hugging Face][11])

## 向量数据库

* **Qdrant（本地模式 / on-disk）**：

    * **Local mode** 支持内存或磁盘持久化，无需独立服务（也可随时切换 Docker
      服务版）。([LangChain][12], [qdrant.tech][13], [Haystack][14])
    * **Named Vectors**：同一 point 持多种向量（例如 `text_dense`、`image`、`text_sparse`
      ），允许有的点只存其一。([qdrant.tech][15])
    * **Sparse/Hybrid**：稀疏向量一等公民，内置 **Hybrid Query**
      （稠密+稀疏服务端融合），官方有混合检索与重排教程。([qdrant.tech][16])
    * **存储**：支持 in-memory 与 on-disk payload，RocksDB 驱动，利于大 payload；可建索引优化过滤。([qdrant.tech][17])

---

# 三、数据与集合设计（Qdrant）

## Collection：`rag_multimodal`

* **named vectors**

    * `text_dense`：size=1024，distance=cosine（Qwen3-Embedding）。
    * `image`：size=512，distance=cosine（OpenCLIP）。
    * `text_sparse`：稀疏向量（BM25/SPLADE++，Qdrant 原生存储）。([qdrant.tech][15])
* **payload（元数据）**

    * `doc_id`, `chunk_id`, `file_path`, `title`, `lang`, `page`, `bbox`,
    * `version`, `sha256`, `mtime`, `phash`(图像), `modality` ∈ {text,image},
    * `caption`（BLIP 英文描述，可选中文翻译）。

> 说明：Qdrant 允许某个 point 只携带其拥有的向量（如图片仅有 `image`
> 向量），这非常适合“文本+图片”同库管理。([qdrant.tech][18])

---

# 四、离线“增量更新”流水线（Scan → Chunk → Embed → Upsert）

1. **扫描**：遍历文件/图片，计算 `sha256` + `mtime`（图片还算 `pHash`）；决定新增/更新/软删。
2. **切片**：

    * 文本/PDF：用 **Unstructured**（`auto/hi_res/ocr_only` 自适应）+ **PyMuPDF** 获取页码/块与可选
      `bbox`。([docs.unstructured.io][19], [pymupdf.readthedocs.io][20])
3. **向量化**：

    * 文本 Dense：Qwen3-Embedding-0.6B（文档用默认指令；查询侧启用 **instruction-aware** 模板）。([Hugging Face][5])
    * 稀疏：**FastEmbed** 生产 BM25 / SPLADE++
      稀疏向量（无外部服务）。([GitHub][21], [qdrant.tech][22], [qdrant.github.io][23])
    * 图片 Dense：OpenCLIP；并用 BLIP 生成 `caption` 入库。([GitHub][9], [Hugging Face][11])
4. **Upsert**：主键 `doc_id#chunk_id#version` 幂等写入；旧版本软删以支持回滚。

---

# 五、检索与重排（Hybrid→Rerank）

* **Hybrid 检索（Qdrant）**：

    * 文本查询：dense(`text_dense`) + sparse(`text_sparse`) 融合；
    * 图像/跨模态：dense(`image`) 与文本 caption 混合；
    * 过滤：按 `modality/lang/time_range` 等 payload。([qdrant.tech][16])
* **重排**：Qwen3-Reranker-0.6B seq-cls，对 TopK≈80 候选重排，取 8–12 进入 LLM 上下文。([Hugging Face][8])

---

# 六、对话历史（短期记忆）与长期“持久记忆”

## A) 对话历史（SQLite）

* **原始日志**：`chat_messages(id, session_id, role, ts, content, token_count, meta)`。
* **滚动摘要**：`chat_summaries(session_id, upto_msg_id, summary_md, tokens)`。
* **策略**：当历史 > \~1800 tokens 或每 5–8 轮触发“**滚动压缩**”；近邻选择保留最近 8 轮里“高信息密度”片段，必要时做句级摘要。
* **投喂顺序**：`Conversation summary`（≤600）→ `Recent turns`（≤400）。

## B) 长期记忆（Qdrant 小集合 + SQLite 索引）

* **集合**：`agent_memory`（`text_dense` + `text_sparse`）。
* **字段**：`kind(profile|preference|fact|project)`, `statement`, `tags[]`, `source_session/msg_ids`, `confidence`,
  `ttl/null`, `decay_score`。
* **检索**：用“当前问题 + 主题关键词”查询，Hybrid 召回 Top-5（总 ≤250 tokens）。
* **记忆门（memory\_gate）**：

    * 规则优先（“从现在开始/以后都…”等）；
    * 轻量分类（长期性、可验证、无明确时效）全满足才入库；
    * 去重合并 + 衰减冷藏（长期未命中降低 `decay_score`）。

> 语言检测用 **fastText lid.176**（176 语种，轻量可靠）；需要时可切 Lingua。([fasttext.cc][24])

---

# 七、上下文拼装与 Token 预算（不轻易裁剪，先聪明压缩）

* **建议预算（可动态调节）**

    * 系统/工具指令 ≤ 400
    * 短期记忆（摘要+近邻）≤ 1 000
    * 长期记忆 ≤ 250
    * 证据上下文 ≤ 2 200
    * 安全缓冲 ≥ 150
* **动态算法**：先按信息价值分配配额→去冗余→句级/段级摘要→摘要再摘要→**最后**才按时间淘汰最老近邻。

---

# 八、设备自适应与量化

* **优先级**：GPU → **重排 + 生成**；嵌入可 CPU 批处理，GPU 空闲再迁移。
* **Transformers 路线**：`bitsandbytes` **8-bit/4-bit** 量化 + `device_map="auto"`（自动把权重分配到可用 GPU/CPU/磁盘），OOM
  自动降批/降精度/切 CPU。([Hugging Face][25])
* **Embedding 服务化（可选）**：Hugging Face **TEI** 支持 **Qwen3-Embedding-0.6B**，CPU/GPU
  皆可运行。([GitHub][26], [Hugging Face][27])

---

# 九、检索/生成参数

* **切片**：每块 640 tokens，overlap 128。
* **检索**：`dense_k=60`，`sparse_k=60`，Hybrid 权重 λ=0.5 起步（用验证集再调）。([qdrant.tech][16])
* **图像检索**：`image_k=40`。
* **重排**：`TopK_in=80 → TopR=10`。
* **答复语言**：跟随用户输入语种；跨语问题保留原文证据 + 目标语摘要。
* **嵌入维度**：1024（与 `text_dense` 保持一致）。

---

# 十、LangGraph 拓扑（关键节点）

```
user_input
  └─► device_resolver
      └─► query_normaliser (含语言检测 / Qwen3 query 指令)
          ├─► history_loader + history_compactor (SQLite)
          ├─► memory_retriever (Qdrant agent_memory, hybrid)
          │     └─► memory_gate + memory_inserter (异步/回合后)
          ├─► retriever_hybrid (Qdrant rag_multimodal)
          ├─► rerank (Qwen3-Reranker-0.6B seq-cls)
          ├─► budget_orchestrator (动态压缩/配额)
          └─► generator (Qwen3-1.7B)
                └─► answer_with_citations (强制证据链接)
```

* **持久化**：LangGraph **SQLite checkpointer**
  保存有向图状态/线程（支持回放/容错/“记忆”式能力）。([LangChain AI][1], [PyPI][28])

---

# 十一、对话与证据输出规范

* **Answer** 正文后附 **Evidence** 列表：

    * 文本：`[标题或文件名](file:///…/paper.pdf#page=12) — 片段摘要`
    * 图片：`[img_00123.jpg](file:///…/img/img_00123.jpg) — caption/位置`
    * 同文件多片段合并展示，避免刷屏。
* 若无足够证据：明确说明“未检到支持性证据”，并返回可复现的检索关键词/过滤条件。

---

# 十二、运行与运维要点

* **Qdrant 本地模式**：可 `path=/your/db` 落盘；需要升级时换 **Docker** 服务版（挂载 `/qdrant/storage`
  ）即可，数据可迁移。([LangChain][12], [qdrant.tech][29])
* **稀疏/混合**：FastEmbed 生产 BM25/SPLADE++；Qdrant
  端直接融合分数；必要时在客户端做归一化/调权。([GitHub][21], [qdrant.tech][22])
* **PDF/版面**：Unstructured `auto/hi_res/ocr_only` 按文件特性自适应；PyMuPDF 提取文本块/页码/可选
  bbox。([docs.unstructured.io][19], [pymupdf.readthedocs.io][20])
* **语言检测**：fastText lid.176（176 语种），轻量、成熟。([fasttext.cc][24])
* **可观测**：最小化日志（SQLite）记录：召回/重排得分、token 预算、内存命中与写入；UI 提供“🧠 记忆抽屉”审计/编辑。

---

# 十三、已处理的关键权衡

* **同源协同**：生成/嵌入/重排均选 Qwen3 家族，语言与风格一致；同时用 OpenCLIP
  解决跨模态检索。([Qwen][30], [Hugging Face][10])
* **简洁 vs 能力**：Qdrant 的 **Local mode + Named Vectors + Hybrid**
  在“文本+图片+混合检索+离线增量”上一次到位，避免多库/多进程复杂性。([qdrant.tech][13])
* **性能 vs 资源**：GPU 优先重排/生成；嵌入批处理可 CPU 跑；必要时以 bitsandbytes + device\_map
  自动分配权重。([Hugging Face][25])
* **Token 成本**：短期历史滚动摘要 + 近邻摘录；长期记忆小集合检索；“先压缩后裁剪”，最大化有效信息。

---

# 十四、补充：极简配置草案（摘）

```yaml
app:
  device: auto      # auto / cpu / cuda:0
models:
  generator: Qwen/Qwen3-1.7B-Base
  embed_text: Qwen/Qwen3-Embedding-0.6B
  reranker: tomaarsen/Qwen3-Reranker-0.6B-seq-cls
  clip_image: laion/CLIP-ViT-B-32-laion2B-s34B-b79K
  captioner: Salesforce/blip-image-captioning-base
vectorstore:
  kind: qdrant_local
  path: ./qdrant_db
  collection: rag_multimodal
  named_vectors:
    text_dense: { size: 1024, distance: cosine }
    image: { size: 512, distance: cosine }
    text_sparse: { sparse: true }
retrieval:
  dense_k: 60
  sparse_k: 60
  hybrid_lambda: 0.5
  image_k: 40
  rerank_topk: 80
  rerank_topr: 10
ingest:
  text_chunk_tokens: 640
  text_chunk_overlap: 128
  caption_lang: en
output:
  require_citations: true
```

---

# 参考链接

* **LangGraph 持久化 / SQLite checkpointer**：官方概念与发布说明。([LangChain AI][1], [LangChain Blog][2])
* **Qwen3-1.7B-Base 及 32K 上下文**：模型卡 & 官方博客。([Hugging Face][3], [Qwen][4])
* **Qwen3-Embedding（0.6B；MRL；Instruction-aware）**：模型卡/仓库/论文；TEI
  部署文档。([Hugging Face][5], [GitHub][6], [arXiv][7])
* **Qwen3-Reranker 序列分类变体**：Hugging Face 模型页。([Hugging Face][8])
* **OpenCLIP ViT-B/32**：仓库与模型卡。([GitHub][9], [Hugging Face][31])
* **BLIP Caption 模型与用法**：模型卡与 Transformers 文档。([Hugging Face][11])
* **Qdrant：Named Vectors / Sparse / Hybrid**：概念文档、Hybrid Queries、Points 说明。([qdrant.tech][15])
* **Qdrant：Local mode（内存/磁盘）与本地落盘示例**：LangChain 集成文档与 Qdrant 指南。([LangChain][12], [qdrant.tech][13])
* **Qdrant：存储策略（InMemory/OnDisk/RocksDB）**。([qdrant.tech][17])
* **FastEmbed（BM25/SPLADE++ 稀疏）**：官方与教程。([GitHub][21], [qdrant.tech][22], [qdrant.github.io][23])
* **Unstructured / PyMuPDF**：分区策略与文本/页码/块提取。([docs.unstructured.io][19], [pymupdf.readthedocs.io][20])
* **语言识别 fastText lid.176**：官方文档。([fasttext.cc][24])
* **Bitsandbytes 量化与 device\_map='auto'**：Transformers/Accelerate 文档。([Hugging Face][25])

[1]: https://langchain-ai.github.io/langgraph/concepts/persistence/?utm_source=chatgpt.com "LangGraph persistence - GitHub Pages"

[2]: https://blog.langchain.com/langgraph-v0-2/?utm_source=chatgpt.com "LangGraph v0.2: Increased customization with new ..."

[3]: https://huggingface.co/Qwen/Qwen3-1.7B-Base?utm_source=chatgpt.com "Qwen/Qwen3-1.7B-Base"

[4]: https://qwenlm.github.io/blog/qwen3/?utm_source=chatgpt.com "Qwen3: Think Deeper, Act Faster | Qwen"

[5]: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B?utm_source=chatgpt.com "Qwen/Qwen3-Embedding-0.6B"

[6]: https://github.com/QwenLM/Qwen3-Embedding?utm_source=chatgpt.com "QwenLM/Qwen3-Embedding"

[7]: https://arxiv.org/pdf/2506.05176?utm_source=chatgpt.com "Qwen3 Embedding: Advancing Text Embedding and ..."

[8]: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls?utm_source=chatgpt.com "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"

[9]: https://github.com/mlfoundations/open_clip?utm_source=chatgpt.com "mlfoundations/open_clip: An open source implementation ..."

[10]: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K?utm_source=chatgpt.com "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

[11]: https://huggingface.co/Salesforce/blip-image-captioning-base?utm_source=chatgpt.com "Salesforce/blip-image-captioning-base"

[12]: https://python.langchain.com/docs/integrations/vectorstores/qdrant/?utm_source=chatgpt.com "Qdrant"

[13]: https://qdrant.tech/documentation/frameworks/langchain/?utm_source=chatgpt.com "Langchain"

[14]: https://haystack.deepset.ai/integrations/qdrant-document-store?utm_source=chatgpt.com "Qdrant - Haystack - Deepset"

[15]: https://qdrant.tech/documentation/concepts/collections/?utm_source=chatgpt.com "Collections"

[16]: https://qdrant.tech/documentation/concepts/hybrid-queries/?utm_source=chatgpt.com "Hybrid Queries"

[17]: https://qdrant.tech/documentation/concepts/storage/?utm_source=chatgpt.com "Storage"

[18]: https://qdrant.tech/documentation/concepts/points/?utm_source=chatgpt.com "Points"

[19]: https://docs.unstructured.io/open-source/core-functionality/partitioning?utm_source=chatgpt.com "Partitioning"

[20]: https://pymupdf.readthedocs.io/en/latest/recipes-text.html?utm_source=chatgpt.com "Text - PyMuPDF 1.26.3 documentation"

[21]: https://github.com/qdrant/fastembed?utm_source=chatgpt.com "qdrant/fastembed: Fast, Accurate, Lightweight Python ..."

[22]: https://qdrant.tech/documentation/fastembed/fastembed-splade/?utm_source=chatgpt.com "Working with SPLADE"

[23]: https://qdrant.github.io/fastembed/examples/SPLADE_with_FastEmbed/?utm_source=chatgpt.com "SPLADE with FastEmbed"

[24]: https://fasttext.cc/docs/en/language-identification.html?utm_source=chatgpt.com "Language identification"

[25]: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes"

[26]: https://github.com/huggingface/text-embeddings-inference?utm_source=chatgpt.com "huggingface/text-embeddings-inference: A blazing fast ..."

[27]: https://huggingface.co/docs/text-embeddings-inference/en/local_cpu?utm_source=chatgpt.com "Using TEI locally with CPU"

[28]: https://pypi.org/project/langgraph-checkpoint-sqlite/?utm_source=chatgpt.com "langgraph-checkpoint-sqlite"

[29]: https://qdrant.tech/documentation/quickstart/?utm_source=chatgpt.com "Local Quickstart"

[30]: https://qwenlm.github.io/blog/qwen3-embedding/?utm_source=chatgpt.com "Advancing Text Embedding and Reranking Through ... - Qwen"

[31]: https://huggingface.co/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k?utm_source=chatgpt.com "laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k"

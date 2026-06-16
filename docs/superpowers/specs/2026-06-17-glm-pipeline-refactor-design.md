# GLM-OCR 工作流重构与 MinerU 优势融合设计规格说明书

本设计文档旨在清理之前侵入式修改 MinerU 库的融合代码，并将 MinerU 对 PDF 预处理、后处理中具备明显优势的 **VLM 流程设计** 融合至我们自研的 `GLM-OCR` 工作流中。同时，阐述如果通过 Dify 等工作流调用该服务的集成架构。

---

## 1. 任务目标

1. **彻底清理融合代码**：物理删除 `GLM-OCR` 下的 `mineru_integration/` 目录，撤销对外部 `mineru` 环境的 Monkey Patch。
2. **Dify 集成工作流**：给出 Dify 通过 HTTP 请求节点调用 MinerU 同步解析 API 提取 Markdown 的集成指南。
3. **融合 MinerU 的预/后处理优势，完善 GLM-OCR 管线**：
   * **中介 JSON（Middle JSON）与本地缓存机制**：规范化输出产物目录，生成并保存包含所有 Layout 框、OCR 内容的 `{pdf_name}_middle.json`。支持本地缓存，避免重复请求扣除 Token。
   * **表格与公式的截图保护（后处理）**：当 Layout 检测到复杂的 `table` 或 `formula` 块时，可根据选项直接在原图上进行区域裁剪保存为截图，插入 Markdown，杜绝排版乱码。
   * **英文字符与连字符智能清洗（后处理）**：移植 MinerU 的 `is_hyphen_at_line_end` 判断算法与 `full_to_half_exclude_marks` 全角英数字转半角算法，消除跨页/跨行文本拼接造成的乱码与符号不兼容。

---

## 2. 详细设计

### 2.1 清理融合代码（物理删除）

* 目标路径：`E:/project/GLM-OCR/mineru_integration/`
* 操作：完全移除该目录（包含 `mineru_glm.py`、`test_mineru_glm.py` 及其下属文件）。
* 确保本地环境的 `mineru` 使用原生代码运行。

---

### 2.2 Dify 工作流集成（以 MinerU FastAPI 服务为例）

Dify 的“HTTP 请求”节点可以无缝对接 MinerU 原生的 FastAPI 服务。

#### 接口地址及 Header
* **请求方法**：`POST`
* **接口 URL**：`http://<MinerU-Host-IP>:8000/file_parse`
* **Content-Type**：`multipart/form-data`

#### 请求参数 (FormData)
* `files`：File 类型，绑定 Dify 流程中的 PDF 文件对象。
* `parse_method`：Text 类型，设为 `auto` (智能解析) 或 `ocr` (强制 OCR)。
* `backend`：Text 类型，设为 `vlm` (强制使用 VLM 模式)。
* `is_only_to_markdown`：Text 类型，设为 `True` (仅输出 markdown)。

#### Dify 下游数据提取 (JSON Path)
从响应 JSON 中提取纯 Markdown 文本：
* **表达式**：`$.data.markdown`

---

### 2.3 GLM-OCR 管线与后处理改进设计

#### A. 目录与中介 JSON 规范化

所有输出文件将统一收纳在以文档主文件名命名的专属子目录下：
```
ocr_output/<PDF主文件名>/
├── <PDF主文件名>.md              # 拼装排版后的 Markdown (含表格/图表/公式截图链接)
├── <PDF主文件名>.docx            # 转换为 Word 的排版产物
├── <PDF主文件名>_middle.json     # 结构化中介 JSON，保存 Layout 与 OCR 结果
└── images/                     # 专属插图、表格截图目录
```

在 `pipeline/orchestrator.py` 运行结束后，除返回 Markdown 字符串外，还将所有的版面检测框、坐标、标签和对应的 OCR 文字/截图路径序列化保存到 `_middle.json`。

#### B. 黄金 JSON 缓存读取逻辑
在 `run_pipeline.py` 开始执行时，先检测 `ocr_output/<PDF主文件名>/<PDF主文件名>_middle.json` 是否存在：
* 如果存在，且没有传递 `--force` 强制刷新参数，则直接读取本地中介 JSON 进行 Markdown 和 Docx 重组排版，**完全不请求任何大模型 API**。
* 如果不存在，或传入了 `--force`，则照常启动本地 Layout 模型与远程 GLM-OCR API 解析。

#### C. 表格 (Table) 与公式 (Formula) 截图化保护
在 `pipeline/orchestrator.py` 的重排序和裁剪逻辑中，我们将提供可选参数（默认开启 `table` 截图）：
* 如果启用表格截图（`table_as_image=True`），当 PP-DocLayout-V3 识别出 `lbl_lower == "table"` 时，**不进行 OCR 识别，而是直接裁剪保存为截图 `page_xxx_table_x.png`** 并在 Markdown 中插入 `![table](images/...)` 占位，以 100% 还原排版。
* 长公式 `formula` 也遵循此逻辑。

#### D. 英文字符与连字符智能清洗
在段落合并重排（`postprocessing.py` 的 `smart_reflow_markdown`）以及跨页缝合（`stitch_pages`）逻辑中引入：
1. **行尾连字符清洗**：识别 `-\u00ad\u2010\u2011\u2043` 结尾的单词断字，自动进行拼合（如 `infor-` 和 `mation` 合并为 `information`）。
2. **全角转半角优化**：移植 `full_to_half_exclude_marks` 将全角英文字母（如 `Ａ` 到 `Ｚ`）和数字规范化为 ASCII 半角（如 `A` 到 `Z`），防止大模型处理文本时遇到未预期的编码异常。

---

## 3. 验证与回归方案

1. **代码回归**：清理完融合代码后，验证 MinerU CLI 原版能否正常独立执行，不受补丁污染影响。
2. **管线单元测试**：
   * 创建测试用 PDF 文件（包含页眉页脚、连字符换行、复杂表格与长公式）。
   * 运行重构后的 `run_pipeline.py`：
     - 验证表格和插图是否全部被干净地裁剪到 `images/` 下。
     - 验证 Markdown 里有无乱码全角字符和折断的连字符。
     - 验证二次解析时是否触发本地 `_middle.json` 缓存直接生成 Markdown。

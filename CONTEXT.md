# GLM-OCR Context

融合高精度版面分析与本地秒级缓存的端到端多模态文档解析管线，旨在实现前处理（版面分析、纠偏、遮罩、XY-Cut）与后处理（连字符清洗、半角规范化、表格截图替换）的完全解耦与模块化设计。

## Language

**LayoutPredictor**:
本地版面检测预测器单例，负责加载 PP-DocLayout-V3 权重，分析图像版面，输出元素检测框（Boxes）与类别标签。在 CI 测试中该模块被 Mock 化，以隔离物理大模型加载开销。
_Avoid_: layout model runner, model detector

**StreamAssembler**:
保序组装与流式写盘器，负责多页 PDF 异步解析中的无序并发 OCR 结果回填、并发幂等防御，并按照物理页面顺序滑动窗口写入 Markdown 磁盘文件与中介缓存。
_Avoid_: page combiner, result writer

**GLM-OCR API**:
远程或本地大模型 VLM 服务接口（通常监听 8700 端口），提供对文本、公式图像的 OCR 识别支持。
_Avoid_: vllm client, vlm completions api

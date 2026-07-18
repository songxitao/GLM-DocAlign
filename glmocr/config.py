import os

# 布局分析模型本地路径
LOCAL_LAYOUT_MODEL = os.getenv(
    "LOCAL_LAYOUT_MODEL",
    r"E:\project\GLM-OCR\model\PP-DocLayoutV3safetensor"
)

# vLLM API 的 Chat Completions 地址
VLLM_API_URL = os.getenv(
    "VLLM_API_URL",
    "http://127.0.0.1:8700/v1/chat/completions"
)

# 模型名称
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "glm-ocr"
)

# 动态派生的模型列表 API 地址，用于服务存活自检
VLLM_API_MODELS_URL = VLLM_API_URL.replace("/v1/chat/completions", "/v1/models")

import sys
import os

# 1. 设置环境变量，指向本地运行的 GLM-OCR vLLM 服务器
os.environ["MINERU_VL_SERVER"] = "http://127.0.0.1:8700"
os.environ["MINERU_VL_API_KEY"] = "dummy_key"

# 2. 动态挂载本地包，这样导入 transformers 和 mineru_vl_utils 时会使用本地集成的版本
# 这能完美覆盖子进程（Windows spawn 方式启动）的导入，从源头上完成 MinerUClient 劫持
integration_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, integration_dir)

# 3. 调用 MinerU CLI 原版入口
from mineru.cli.client import main

if __name__ == "__main__":
    # 动态将 --backend 默认参数追加到 argv 里，默认使用混合 API 识别模式（本地 Layout + 远程 GLM-OCR）
    if not any(arg in sys.argv for arg in ['-b', '--backend']):
        sys.argv.extend(['--backend', 'vlm-http-client'])
        
    print("[GLM-OCR Integration] 正在以 GLM-OCR 集成模式启动 MinerU CLI ...")
    main()

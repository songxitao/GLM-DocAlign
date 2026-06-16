import sys
import os
import torch
import importlib.util
from PIL import Image

# 1. 设置环境变量，指向本地运行的 GLM-OCR vLLM 服务器
# 这将覆盖 MinerU 对远程 VLM 服务器的 API 请求目标
os.environ["MINERU_VL_SERVER"] = "http://127.0.0.1:8700"
os.environ["MINERU_VL_API_KEY"] = "dummy_key"

# 2. 动态挂载本地魔改版 transformers（包含 pp_doclayout_v3模型定义）
# 这可以防止 mineru 环境下因 transformers 版本代差而无法识别 pp_doclayout_v3 架构的问题
integration_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, integration_dir)

import transformers
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# 3. 延迟加载 Layout 模型，只有在真正需要版面分析时才初始化，避免单元测试时也加载模型
layout_model = None
layout_processor = None

def load_layout_model_lazy():
    global layout_model, layout_processor
    if layout_model is None:
        LAYOUT_MODEL_PATH = r"E:\project\GLM-OCR\model\PP-DocLayoutV3safetensor"
        print(f"[Init] 正在加载本地版面分析模型: {LAYOUT_MODEL_PATH} ...")
        # 强制在 CPU 上运行版面分析，极速且不占用大模型的 GPU 显存
        layout_model = AutoModelForObjectDetection.from_pretrained(LAYOUT_MODEL_PATH).to("cpu")
        layout_processor = AutoImageProcessor.from_pretrained(LAYOUT_MODEL_PATH)
        print("[Init] 版面模型加载就绪。")

def map_and_normalize_box(label: str, box: list[float], width: int, height: int) -> tuple[str, list[float]]:
    # 标签映射：'formula' -> 'equation', 'figure' -> 'image', 其它转为小写
    label_lower = label.lower()
    if label_lower == 'formula':
        mapped_label = 'equation'
    elif label_lower == 'figure':
        mapped_label = 'image'
    else:
        mapped_label = label_lower

    # 坐标归一化，范围限制在 [0, 1]
    w = max(1, width)
    h = max(1, height)

    x1 = max(0.0, min(1.0, float(box[0]) / w))
    y1 = max(0.0, min(1.0, float(box[1]) / h))
    x2 = max(0.0, min(1.0, float(box[2]) / w))
    y2 = max(0.0, min(1.0, float(box[3]) / h))

    return mapped_label, [x1, y1, x2, y2]

# 4. 编写劫持 batch_layout_detect 的逻辑
def my_batch_layout_detect(self, images, priority=None, scored=None):
    load_layout_model_lazy()
    results = []
    print(f"[Layout] 正在通过 PP-DocLayout-V3 进行版面分析，共 {len(images)} 页...")
    
    from mineru_vl_utils.structs import ContentBlock, ExtractResult
    
    for image in images:
        width, height = image.size
        # 用 CPU 预测版面方块
        inputs = layout_processor(images=image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = layout_model(**inputs)
            
        raw_results = layout_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])[0]
        
        blocks = []
        for score, label_id, box in zip(raw_results["scores"], raw_results["labels"], raw_results["boxes"]):
            if score.item() < 0.4:
                continue
            
            label_name = layout_model.config.id2label.get(label_id.item(), f"Label_{label_id.item()}")
            lbl_lower = label_name.lower()
            if lbl_lower in ["header", "footer"]:
                continue  # 过滤掉页眉页脚
                
            mapped_label, nbox = map_and_normalize_box(lbl_lower, box.tolist(), width, height)
            
            # 构建标准的 ContentBlock
            block = ContentBlock(type=mapped_label, bbox=nbox)
            blocks.append(block)
            
        results.append(ExtractResult(blocks))
    return results

# 5. 编写单页/异步的劫持逻辑，完全路由到已有的批量预测中
def my_layout_detect(self, image, priority=None, scored=None):
    results = my_batch_layout_detect(self, [image], priority, scored)
    return results[0]

async def my_aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
    results = my_batch_layout_detect(self, [image], priority, scored)
    return results[0]

async def my_aio_batch_layout_detect(self, images, priority=None, semaphore=None, scored=None):
    # 异步版本的劫持，在 CPU 上同步跑完并直接返回
    return my_batch_layout_detect(self, images, priority, scored)

# 6. 实施 Monkey Patch 劫持 MinerUClient 的全部 4 个版面预测入口
from mineru_vl_utils import MinerUClient
MinerUClient.layout_detect = my_layout_detect
MinerUClient.aio_layout_detect = my_aio_layout_detect
MinerUClient.batch_layout_detect = my_batch_layout_detect
MinerUClient.aio_batch_layout_detect = my_aio_batch_layout_detect
print("[Patch] 成功全面劫持 MinerUClient (layout_detect/aio_layout_detect/batch_layout_detect/aio_batch_layout_detect)。")

# 7. 调用 MinerU CLI 原版入口
from mineru.cli.client import main

if __name__ == "__main__":
    # 动态将 --backend 默认参数追加到 argv 里，默认请求远程/本地的 VLM 服务
    if not any(arg in sys.argv for arg in ['-b', '--backend']):
        sys.argv.extend(['--backend', 'vlm-http-client'])
        
    print("[GLM-OCR Integration] 正在以 GLM-OCR 集成模式启动 MinerU CLI ...")
    main()

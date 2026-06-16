import os
import sys

# 将物理路径指向真正的 site-packages 下的 mineru_vl_utils
# 这样它就会从真实的包路径中加载所有的子模块 (mineru_client, structs 等)
real_path = r"E:\conda\envs\mineru\lib\site-packages\mineru_vl_utils"
__path__ = [real_path]

# 正常从原版中导入我们需要劫持的类和结构
from mineru_vl_utils.mineru_client import MinerUClient
from mineru_vl_utils.structs import ContentBlock, ExtractResult

# ================= 实施 MONKEY PATCH 劫持 =================
import torch
from PIL import Image

layout_model = None
layout_processor = None

def load_layout_model_lazy():
    global layout_model, layout_processor
    if layout_model is None:
        LAYOUT_MODEL_PATH = r"E:\project\GLM-OCR\model\PP-DocLayoutV3safetensor"
        print(f"[Init] 正在加载本地版面分析模型: {LAYOUT_MODEL_PATH} ...")
        # 强制使用本地集成的高版本 transformers
        import transformers
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        layout_model = AutoModelForObjectDetection.from_pretrained(LAYOUT_MODEL_PATH).to("cpu")
        layout_processor = AutoImageProcessor.from_pretrained(LAYOUT_MODEL_PATH)
        print("[Init] 版面模型加载就绪。")

def map_and_normalize_box(label: str, box: list[float], width: int, height: int) -> tuple[str, list[float]]:
    label_lower = label.lower()
    if label_lower == 'formula':
        mapped_label = 'equation'
    elif label_lower == 'figure':
        mapped_label = 'image'
    else:
        mapped_label = label_lower
    w = max(1, width)
    h = max(1, height)
    x1 = max(0.0, min(1.0, float(box[0]) / w))
    y1 = max(0.0, min(1.0, float(box[1]) / h))
    x2 = max(0.0, min(1.0, float(box[2]) / w))
    y2 = max(0.0, min(1.0, float(box[3]) / h))
    return mapped_label, [x1, y1, x2, y2]

def my_batch_layout_detect(self, images, priority=None, scored=None):
    load_layout_model_lazy()
    results = []
    print(f"[Layout] 正在通过 PP-DocLayout-V3 进行版面分析，共 {len(images)} 页...")
    for image in images:
        width, height = image.size
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
                continue
            mapped_label, nbox = map_and_normalize_box(lbl_lower, box.tolist(), width, height)
            block = ContentBlock(type=mapped_label, bbox=nbox)
            blocks.append(block)
        results.append(ExtractResult(blocks))
    return results

def my_layout_detect(self, image, priority=None, scored=None):
    results = my_batch_layout_detect(self, [image], priority, scored)
    return results[0]

async def my_aio_layout_detect(self, image, priority=None, semaphore=None, scored=None):
    results = my_batch_layout_detect(self, [image], priority, scored)
    return results[0]

async def my_aio_batch_layout_detect(self, images, priority=None, semaphore=None, scored=None):
    return my_batch_layout_detect(self, images, priority, scored)

# 实施方法覆盖
MinerUClient.layout_detect = my_layout_detect
MinerUClient.aio_layout_detect = my_aio_layout_detect
MinerUClient.batch_layout_detect = my_batch_layout_detect
MinerUClient.aio_batch_layout_detect = my_aio_batch_layout_detect
print("[GLM-Source-Patch] 成功在导入源头（mineru_vl_utils）完成了 MinerUClient 劫持。")

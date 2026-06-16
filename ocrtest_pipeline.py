# ocrtest_pipeline.py (GLM-OCR + Layout 极简离线功能测试脚本)

import os
import sys
import base64
import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# 配置 sys.stdout/stderr 防止 Windows 终端中文乱码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 1. 配置路径
LOCAL_LAYOUT_MODEL = r"E:\project\GLM-OCR\model\PP-DocLayoutV3safetensor"
VLLM_API_URL = "http://127.0.0.1:8700/v1/chat/completions"
MODEL_NAME = "glm-ocr"

def test_layout_model(image_path: str):
    """
    测试本地 PP-DocLayoutV3 (safetensors版本) 的版面分析能力，并生成画框诊断图
    """
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片: '{image_path}'，请传入有效路径。")
        return None
        
    print(f"\n[Step 1] 📂 加载测试图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    print(f"🔄 正在加载本地 PP-DocLayoutV3 (safetensors) 模型...")
    # 强制在 CPU 上运行版面分析，只花几十毫秒，绝对不占 GPU 显存
    model = AutoModelForObjectDetection.from_pretrained(LOCAL_LAYOUT_MODEL).to("cpu")
    image_processor = AutoImageProcessor.from_pretrained(LOCAL_LAYOUT_MODEL)
    print("✅ 版面模型加载就绪。")
    
    # 预处理与前向推理
    print("🧠 运行版面分析检测...")
    inputs = image_processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 后处理，将坐标映射还原到原图大小
    results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
    
    # 用 PIL 画框并裁剪
    draw = ImageDraw.Draw(image)
    boxes_detected = []
    
    print("\n🔍 目标检测结果:")
    for result in results:
        for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
            score = score.item()
            if score < 0.4:  # 置信度阈值过滤
                continue
                
            label = label_id.item()
            label_name = model.config.id2label.get(label, f"Label_{label}")
            box_coords = [int(i) for i in box.tolist()] # xmin, ymin, xmax, ymax
            
            print(f"  [{idx + 1}] 类别: {label_name:<10} | 置信度: {score:.2f} | 坐标: {box_coords}")
            
            # 画红色矩形框
            draw.rectangle(box_coords, outline="red", width=3)
            
            boxes_detected.append({
                "index": idx,
                "label": label_name,
                "coords": box_coords
            })
            
    # 保存诊断图
    diag_path = "layout_diagnostic.png"
    image.save(diag_path)
    print(f"\n🎨 画框诊断图已保存至: '{os.path.abspath(diag_path)}' (您可以打开它确认框是否画得准)")
    return boxes_detected

def test_glmocr_api(image_path: str, box_coords: list, label: str):
    """
    裁剪局部图片并发送给本地 vLLM OCR 接口，验证 OCR 的端到端调用
    """
    print(f"\n[Step 2] ✂️ 裁剪局部区域 [{label}] 进行 OCR 测试...")
    image = Image.open(image_path).convert("RGB")
    cropped = image.crop((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
    
    # 保存局部切片
    cropped_path = "crop_temp.png"
    cropped.save(cropped_path)
    
    # 转 base64
    with open(cropped_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
    # 根据类别设定不同的引导词
    prompt = "Text Recognition:"
    if label.lower() == "table":
        prompt = "Table Recognition:"
    elif label.lower() == "formula":
        prompt = "Formula Recognition:"
        
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    print(f"🚀 发送请求至 vLLM ({VLLM_API_URL}) | Prompt: '{prompt}'...")
    try:
        resp = requests.post(VLLM_API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        res_text = resp.json()["choices"][0]["message"]["content"]
        print("\n📝 模型识别响应结果:")
        print("-" * 50)
        print(res_text)
        print("-" * 50)
    except Exception as e:
        print(f"❌ 请求失败: {e}。请确认您已通过 start_glmocr.bat 跑起 vLLM 后端服务！")

if __name__ == "__main__":
    # 接收命令行传入图片，如未传入则默认使用 E:\project\GLM-OCR\my_template.docx 的同级测试图
    test_img = "test_image.png"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        
    if not os.path.exists(test_img):
        # 如果当前不存在 test_image.png，我们提示用户并退出
        print(f"📢 请在命令行指定测试图片路径，例如: python ocrtest_pipeline.py E:\\your_test_image.png")
        sys.exit(0)
        
    # 1. 运行版面分析
    detected_boxes = test_layout_model(test_img)
    
    # 2. 如果检测到了物体，选第一个物体进行 OCR 测试（需先启动 vllm 服务）
    if detected_boxes:
        first_box = detected_boxes[0]
        test_glmocr_api(test_img, first_box["coords"], first_box["label"])
    else:
        print("⚠️ 未在该图片中检测到任何版面框，无法进行下一步局部 OCR 测试。")

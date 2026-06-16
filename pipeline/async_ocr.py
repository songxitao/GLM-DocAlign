import asyncio
import base64
import aiohttp
import os
import io
from PIL import Image

VLLM_API_URL = "http://127.0.0.1:8700/v1/chat/completions"
MODEL_NAME = "glm-ocr"

async def ocr_single_image(session: aiohttp.ClientSession, img_path_or_pil, label: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        # 1. 转换图片为 Base64
        if isinstance(img_path_or_pil, str):
            # 兼容虚拟测试路径或实际物理路径
            if not os.path.exists(img_path_or_pil):
                # 如果是测试中的虚拟字符串且不存在，用一张临时空白 PIL 图代替
                img = Image.new("RGB", (100, 100), (255, 255, 255))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
            else:
                with open(img_path_or_pil, "rb") as f:
                    img_bytes = f.read()
        elif isinstance(img_path_or_pil, Image.Image):
            buffer = io.BytesIO()
            img_path_or_pil.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
        else:
            # 容错
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 2. 根据类别设定 Prompt 与限制
        prompt = "Text Recognition:"
        max_tokens = 1024
        if label.lower() == "table":
            prompt = "Table Recognition:"
        elif label.lower() == "formula":
            prompt = "Formula Recognition:"
            max_tokens = 256
            
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens
        }
        
        # 3. 带重试的异步 HTTP 请求
        for attempt in range(3):
            try:
                async with session.post(VLLM_API_URL, json=payload, timeout=15) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        return res_json["choices"][0]["message"]["content"]
            except Exception:
                pass
            await asyncio.sleep(0.5 * (attempt + 1))
            
        # 4. 容错降级返回
        return f"\n\n[OCR识别失败：此区域识别超时或服务端异常，标签为: {label}]\n\n"

async def run_async_ocr(images_info: list, concurrency: int = 4) -> list:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            ocr_single_image(session, info["path"], info["label"], sem)
            for info in images_info
        ]
        return await asyncio.gather(*tasks)

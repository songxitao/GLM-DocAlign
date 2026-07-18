import pytest
import aiohttp
from unittest.mock import MagicMock

# Monkeypatch aiohttp.ClientResponse.__init__ to support aioresponses in aiohttp >= 3.14.0
orig_init = aiohttp.ClientResponse.__init__
def patched_init(self, *args, **kwargs):
    if 'stream_writer' not in kwargs:
        kwargs['stream_writer'] = MagicMock()
    return orig_init(self, *args, **kwargs)
aiohttp.ClientResponse.__init__ = patched_init

from aioresponses import aioresponses
from glmocr.pipeline.async_ocr import run_async_ocr

@pytest.mark.asyncio
async def test_async_ocr_success_and_fallback():
    with aioresponses() as m:
        # Mock 两个正常请求和一个超时失败请求
        m.post('http://127.0.0.1:8700/v1/chat/completions', status=200, payload={
            "choices": [{"message": {"content": "Recognized Text 1"}}]
        })
        m.post('http://127.0.0.1:8700/v1/chat/completions', exception=aiohttp.ServerTimeoutError())
        
        # 传入两个虚拟图片数据，以及类别
        images_info = [
            {"path": "dummy1.png", "label": "text"},
            {"path": "dummy2.png", "label": "formula"}
        ]
        
        results = await run_async_ocr(images_info, concurrency=2)
        
        assert len(results) == 2
        assert results[0] == "Recognized Text 1"
        # 失败降级断言
        assert "[OCR识别失败" in results[1]

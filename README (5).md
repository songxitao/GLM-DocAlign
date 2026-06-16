---
studios:
- OpenDataLab/MinerU
---

[MinerU](https://github.com/opendatalab/MinerU)项目中使用的模型，欢迎下载使用。
模型使用请参考[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)项目。

### SDK Download

```bash
# First, install the ModelScope library using pip:
pip install modelscope
```

```python
# Use the following Python code to download the model using the ModelScope SDK:
from modelscope import snapshot_download
model_dir = snapshot_download('opendatalab/PDF-Extract-Kit')
```

### Git Download
Alternatively, you can use Git to clone the model repository from ModelScope:

```bash
git clone https://www.modelscope.cn/opendatalab/PDF-Extract-Kit.git
```

---
license: apache-2.0
---

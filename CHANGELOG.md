# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-19

### Added
- **核心包化物理重构 (Core Packaging)**:
  - 新建主包命名空间目录 `glmocr/`，将散落的前处理 `preprocessing.py`、后处理 `postprocessing.py` 和整个 `pipeline/` 目录移入，大幅提升物理局部性（Locality）。
  - 各核心模块与测试用例的导入升级为基于 `glmocr` 的绝对包名导入。
  - 移除测试脚本开头的所有 `sys.path.append(...)` 路径修改。
- **配置外置化与 API 彻底解耦 (Configuration Decoupling)**:
  - 新增 `glmocr/config.py`，集中使用 `os.getenv` 读取系统环境变量，解除所有硬编码，默认值回退支持开箱即用。
  - 升级 `run_pipeline.py`、`appocr_vllm_ui.py`、`appocr_final.py` 以及测试用例，完全对接解耦配置。
- **Mock-First 测试加固与 CI 隔离 (Mock-First Testing & CI)**:
  - 在 `test_orchestrator.py` 中使用 `unittest.mock.patch` 并重构 `MockInputs` 隔离 PP-DocLayout-V3 模型加载，阻止物理模型调用。
  - 运用 `aioresponses` 彻底阻断测试中对远程 VLM 大模型服务的 HTTP 泄漏，实现 15 项测试在无模型和网络下 100% 离线绿灯。
  - 配置 `.github/workflows/pytest-ci.yml` 自动化 CI。使用 `CPU-only torch` 对下载耗时进行了深度调优（体积由 2GB+ 优化为 150MB+）。
- **开源合规与元数据健全 (Legal & Metadata)**:
  - 补齐标准 Apache-2.0 许可证。
  - 新建 `pyproject.toml` 作为现代 PEP 621 元数据标准，配置 setuptools 对非包目录的过滤白名单。
  - 大幅扩充并健全 `.gitignore` 以自动忽略 python 编译、构建缓存及虚拟环境。
  - 物理清理根目录下所有冗余旧 README 复件与废弃启动脚本。

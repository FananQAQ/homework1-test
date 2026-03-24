# 作业说明

## 功能概览

- 基于 `llama.cpp` 源码构建 `local_llm.exe`
- 执行 smoke + bench + 性能 + 准确度测试
- 在 `test_results/` 下输出完整报告与图表

## 快速开始（Windows）

1. 在当前目录打开终端
2. 运行：

```bat
./run_all.bat
```

也可以在资源管理器中直接双击 `run_all.bat`。

## 环境要求

- Windows 10/11 x64
- Git（已加入 PATH）
- CMake >= 3.14（已加入 PATH）
- Python 3.9+（已加入 PATH）
- Visual Studio 2022 C++ 工具链（推荐）

## 目录结构

- `run_all.bat`：一键执行完整流程
- `CMakeLists.txt`：构建入口（生成 `local_llm.exe`）
- `tests/`：评测脚本
- `data/appointment_cert_dataset.jsonl`：200 条数据集
- `docs/`：编译、性能、准确度、技术总结文档
- `lanucher/main.c`：用于双击根目录下的 `local_llm.exe` 即可执行测试

运行后自动生成（已被 `.gitignore` 忽略）：

- `llama.cpp/`（脚本自动克隆）
- `build/`
- `models/*.gguf`
- `test_results/`

## 常用环境变量

- `SKIP_BUILD=1`：跳过编译
- `SKIP_PERF_PY=1`：跳过 `perf_smoke_test.py`
- `SKIP_PIP_MPL=1`：不自动安装 matplotlib
- `SKIP_ACCURACY=1`：跳过准确度评测
- `ACCURACY_LIMIT=20`：只评测前 N 条
- `ACC_N_PREDICT=16`：准确度评测每条最大生成 token（默认 16）
- `ACC_NGL=0`：准确度评测 `ngl`（有 GPU 可试 `999`）
- `SKIP_ROOT_LAUNCHER=1`：跳过生成根目录启动器 `local_llm.exe`

示例（快速调试）：

```bat
set ACCURACY_LIMIT=20
set ACC_N_PREDICT=32
set ACC_NGL=0
run_all.bat
```

## 输出结果

每次运行会创建：

`test_results/run_YYYYMMDD_HHMMSS/`

常见文件：

- `test_report.txt`
- `perf_smoke_*.json`
- `perf_smoke_*.md`
- `perf_smoke_*.png`（安装 matplotlib 时）
- `accuracy_full.json`
- `accuracy_summary.txt`

## 常见问题

- 若 CMake 报错 `nmake` / `CMAKE_C_COMPILER not set`：
  说明本机缺少或未加载 VS C++ 构建工具链，请安装/配置后重试。
- 若准确率相关指标偏低：
  常见原因是模型能力或输出格式不稳定，可增大 `ACC_N_PREDICT`，或替换中文能力更强的 GGUF 模型。

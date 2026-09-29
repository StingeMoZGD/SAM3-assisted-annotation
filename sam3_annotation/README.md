# SAM3 视频目标标注工具（网页版）

基于 [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) 的视频目标跟踪标注工具，采用 Flask 提供网页界面，在浏览器中完成"打点标注 → 模型跟踪 → 修正 → 导出 YOLO 数据集"的完整流程。

## 功能

- 选择 / 上传视频，浏览任意帧
- 在帧上标注正样本点 + 负样本点（每个目标 8 正 + 8 负），支持多目标（自动分配 obj_id）
- 加载 SAM3 模型后运行传播跟踪，显示每帧检测框
- 在画面上拖动修正跟踪框
- 保存 / 加载标注项目（JSON 格式）
- 导出 YOLO 格式数据集（images / labels / classes.txt / config.json）

## 目录结构

```
.
├── app.py                  # Flask 后端（全部逻辑：帧读取、跟踪、导出、项目存取）
├── start_web.sh            # 启动脚本
├── templates/
│   └── index.html          # 前端页面（自包含，无外部 CDN 依赖）
├── model/                  # 模型权重目录（需自行下载，已 gitignore）
│   ├── sam3.pt
│   └── bpe_simple_vocab_16e6.txt.gz
├── videos/                 # 视频目录（上传的视频保存在 videos/uploads/）
├── projects/               # 标注项目保存目录（运行时自动创建）
└── exports/                # YOLO 导出目录（运行时自动创建）
```

## 环境要求

- Linux + NVIDIA GPU（需 CUDA）
- Python 3.10+（本项目验证于 Python 3.12）
- 已安装 [SAM3 官方包](https://github.com/facebookresearch/sam3)
- SAM3 模型权重（见下文获取方式）

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

torch 建议按显卡 CUDA 版本安装官方包，例如：

```bash
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. 安装 SAM3 官方包

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

### 3. 下载模型权重

SAM3 权重需在 Hugging Face 申请访问权限：[facebook/sam3](https://huggingface.co/facebook/sam3)（同意协议并通过后，登录 Hugging Face 下载）。将以下文件放入本仓库 `model/` 目录：

| 文件 | 说明 |
|---|---|
| `sam3.pt` | 模型权重，约 3.3GB（未包含在本仓库中） |
| `bpe_simple_vocab_16e6.txt.gz` | BPE 词表（也可从 SAM3 官方仓库获取） |

## 启动

```bash
# 方式一：激活已装好 sam3 的 conda 环境后直接启动
conda activate sam3
./start_web.sh

# 方式二：指定 Python 解释器
SAM3_PYTHON=/path/to/envs/sam3/bin/python ./start_web.sh

# 方式三：直接用 python 运行
python app.py
```

默认监听 `0.0.0.0:9930`，浏览器打开 `http://<服务器IP>:9930` 即可使用。

### 可配置项

模型权重、BPE 词表、视频目录的取值优先级（从高到低）：

**命令行参数 > 环境变量 > 默认路径（相对仓库根目录）**

| 命令行参数 | 环境变量 | 默认值 |
|---|---|---|
| `--checkpoint` | `SAM3_CHECKPOINT` | `model/sam3.pt` |
| `--bpe` | `SAM3_BPE` | `model/bpe_simple_vocab_16e6.txt.gz` |
| `--videos-dir` | `SAM3_VIDEOS_DIR` | `videos/` |
| `--host` / `--port` | — | `0.0.0.0` / `9930` |

示例：

```bash
./start_web.sh --port 8080 --checkpoint /data/models/sam3.pt
```

## 使用流程

1. 选择视频（扫描 `videos/` 目录）或上传本地视频
2. 在画面上依次标注各目标的正 / 负样本点
3. 选择 GPU，点击"加载模型"
4. 点击"运行跟踪"，等待传播完成；可直接拖动修正检测框
5. 保存标注项目，或导出 YOLO 格式数据集（含 `classes.txt`）

## 注意事项

- 跟踪结果与 YOLO 导出统一使用**归一化相对坐标** `[obj_id, x, y, w, h]`（0~1，x/y 为框左上角）
- app.py 启动时会将 `TMPDIR` 指向仓库内 `_tmp/` 目录（在 import 库之前设置），避免根分区空间不足导致上传临时文件失败；如需其他位置可直接修改代码顶部常量
- 加载模型与运行跟踪需要可用的 CUDA GPU

## 许可证

本工具代码基于 SAM3 构建。SAM3 模型权重与官方代码遵循 [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，下载和使用前请阅读并遵守其条款。

## 致谢

- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI Research
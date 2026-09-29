#!/bin/bash
# SAM3 自动化标注工具（网页版）启动脚本
# 用法: ./start_web.sh [--port 9930] [--host 0.0.0.0] [--checkpoint 权重路径] ...
#
# Python 解释器查找顺序（需已安装 sam3、flask、torch、opencv）：
#   1. 环境变量 SAM3_PYTHON 指定的解释器
#   2. 当前已激活 conda 环境的 python（CONDA_PREFIX）
#   3. python3
cd "$(dirname "$0")"

if [ -n "$SAM3_PYTHON" ] && [ -x "$SAM3_PYTHON" ]; then
  PY="$SAM3_PYTHON"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY="$CONDA_PREFIX/bin/python"
else
  PY="python3"
fi

echo "使用 Python: $PY"
exec "$PY" app.py "$@"
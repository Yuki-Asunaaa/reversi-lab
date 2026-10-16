#!/bin/zsh
# 切换到脚本所在目录，允许从 Finder 双击启动。
cd -- "$(dirname -- "$0")"
exec python3 main.py

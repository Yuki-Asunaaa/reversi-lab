"""统一局面分析与三分类胜率；概率模型只来自项目自身的校准数据。"""

import json
import math
from pathlib import Path

from .board import Board, count
from .search import deterministic

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "calibration.json"


def features(score: float, empty_count: int):
    """评分统一为黑方视角，按阶段加入交互项；截断极端分避免数值溢出。"""
    s = max(-4.0, min(4.0, score / 500.0))
    phase = (60 - empty_count) / 60
    return [1.0, s, s * phase, abs(s), phase, phase * phase]


def softmax(logits):
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def predict(score: float, empty_count: int, model: dict):
    x = features(score, empty_count)
    probabilities = softmax(
        [sum(a * b for a, b in zip(weights, x)) for weights in model["weights"]]
    )
    return dict(zip(("black", "draw", "white"), probabilities))


def analyze(board: Board, seconds=0.4, cancel=None):
    """固定三层分析口径；小残局增加搜索，但只在完全解出时覆盖概率。"""
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else None
    if board.terminal():
        winner = board.winner()
        return {
            "status": "terminal",
            "probabilities": {
                "black": float(winner == 1),
                "draw": float(winner == 0),
                "white": float(winner == -1),
            },
            "score": count(board.black) - count(board.white),
            "depth": 0,
            "exact": True,
            "label": "棋局已结束，显示实际结果",
        }
    # 主概率统一用 depth=3 的评分，避免随主对弈算法或时间档位改变标尺。
    base = deterministic(board, seconds * 0.55, max_depth=3, cancel=cancel)
    if board.empty_count <= 10:
        endgame = deterministic(board, seconds * 0.45, cancel=cancel)
        if endgame.exact:
            black_score = endgame.score * board.turn
            return {
                "status": "solved",
                "probabilities": None,
                "score": black_score,
                "outcome": (black_score > 0) - (black_score < 0),
                "depth": endgame.depth,
                "exact": True,
                "label": "已解残局：双方最佳应对下的结果，不代表玩家不会失误",
            }
    black_score = base.score * board.turn
    calibrated = model is not None and base.depth >= 3
    return {
        "status": "calibrated" if calibrated else "provisional",
        "probabilities": predict(black_score, board.empty_count, model) if calibrated else None,
        "score": black_score,
        "depth": base.depth,
        "nodes": base.nodes,
        "exact": False,
        "label": (
            "参考三层 Alpha-Beta 自对弈水平的估算"
            if calibrated
            else "分析尚未完成或缺少校准模型，暂不显示百分比"
        ),
        "validation": model.get("validation") if model else None,
    }


def analysis_worker(board, queue, cancel):
    try:
        queue.put(analyze(board, cancel=cancel))
    except Exception as error:
        queue.put({"status": "error", "label": str(error), "probabilities": None})

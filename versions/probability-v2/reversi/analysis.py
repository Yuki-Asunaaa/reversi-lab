"""概率 V2：更深的统一局面分析、双方配置条件化、独立标注最优残局结论。"""

import json
from pathlib import Path

from .advanced import advanced
from .board import Board, count
from .probability import features, predict, reference_config, softmax

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "probability_v2.json"


def score_position(board, seconds=0.6, cancel=None):
    """统一 V3 五层评分；同时记录相邻完成层分歧，供模型识别不稳定局面。"""
    completed = []
    result = advanced(
        board,
        seconds,
        version=3,
        max_depth=5,
        cancel=cancel,
        progress=lambda data: completed.append((data["depth"], data["score"])),
    )
    layers = dict(completed)
    previous = layers.get(result.depth - 1, result.score)
    return result, abs(result.score - previous)


def analyze(board: Board, seconds=1.2, cancel=None, config=None):
    """训练与线上使用同一评分器。未知人类和训练范围外的时长明确标出。"""
    if board.terminal():
        winner = board.winner()
        return dict(
            status="terminal",
            probabilities={
                "black": float(winner == 1),
                "draw": float(winner == 0),
                "white": float(winner == -1),
            },
            score=count(board.black) - count(board.white),
            depth=0,
            exact=True,
            model_version=2,
            label="棋局已结束，显示实际结果",
        )
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else None
    # 中前盘五层分析最多 0.6 秒，残局把总预算余量用于精确解算。
    base, disagreement = score_position(board, min(seconds, 0.6), cancel)
    solved = base if base.exact else None
    if solved is None and board.empty_count <= 14:
        remaining = max(0.001, seconds - base.elapsed)
        endgame = advanced(board, remaining, version=3, cancel=cancel)
        if endgame.exact:
            solved = endgame
    black_score = base.score * board.turn
    if solved is not None:
        black_score = solved.score * board.turn
        return dict(
            status="solved",
            probabilities=None,
            score=black_score,
            outcome=(black_score > 0) - (black_score < 0),
            depth=solved.depth,
            exact=True,
            model_version=2,
            label="已解残局：双方最佳应对下的结果，不等于实际玩家必然如此落子",
        )
    sides, notes = reference_config(config)
    available = model is not None and base.depth >= 3
    if model:
        low, high = model["budget_bounds"]
        if any(not low <= float(side["seconds"]) <= high for side in sides.values()):
            notes.append(f"时长超出训练范围 {low}～{high} 秒，按边界时长给出参考估计")
        context_key = "|".join((sides["black"]["algorithm"], sides["white"]["algorithm"]))
        if context_key not in model["training"]["algorithm_pairs"]:
            notes.append("此算法配对未直接用于训练")
    if base.depth < 5:
        notes.append(f"预算内仅完成 {base.depth} 层")
    label = "V2 条件化局面估计，尚未进行独立准确性验证"
    if not available:
        label = "分析不足或尚未生成 V2 模型，暂不显示百分比"
    if notes:
        label += "；" + "；".join(notes)
    return dict(
        status="estimated" if available else "provisional",
        probabilities=(
            predict(black_score, board.empty_count, model, base.depth, disagreement, config)
            if available
            else None
        ),
        score=black_score,
        depth=base.depth,
        nodes=base.nodes,
        exact=False,
        disagreement=disagreement,
        model_version=2,
        validation=None,
        reference=sides,
        label=label,
    )


def analysis_worker(board, config, queue, cancel):
    """分析使用回合开始前的配置快照，不读取对弈进程中的可变状态。"""
    try:
        queue.put(analyze(board, cancel=cancel, config=config))
    except Exception as error:
        queue.put({"status": "error", "label": str(error), "probabilities": None})

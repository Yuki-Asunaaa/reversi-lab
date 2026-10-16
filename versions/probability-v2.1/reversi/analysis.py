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
    # 主栏固定双方参考策略，绝不把当前棋手身份混进棋盘局势。
    neutral = {
        "mode": "eve",
        "black": {"algorithm": "hybrid_v3", "seconds": 0.2},
        "white": {"algorithm": "hybrid_v3", "seconds": 0.2},
    }
    sides, notes = reference_config(config)
    available = model is not None and base.depth >= 3
    opening_uncovered = board.empty_count > 54
    position_available = available and not opening_uncovered
    match_available = position_available
    if opening_uncovered:
        notes.append("初始及前六步局面未被训练样本覆盖，暂不显示百分比")
    if model:
        low, high = model["budget_bounds"]
        if any(not low <= float(side["seconds"]) <= high for side in sides.values()):
            match_available = False
            notes.append(f"当前时限超出训练范围 {low}～{high} 秒，暂不提供对战百分比")
        context_key = "|".join((sides["black"]["algorithm"], sides["white"]["algorithm"]))
        if context_key not in model["training"]["algorithm_pairs"]:
            match_available = False
            notes.append("此算法配对未直接用于训练，暂不提供对战百分比")
    if config and config.get("mode") == "pve":
        match_available = False
        notes = [note for note in notes if not note.startswith("人类棋力未知")]
        notes.insert(0, "人类棋力未知，暂不提供人机对战百分比；可参考上方局势估计")
    label = "固定双方 V3、0.2 秒参考条件；交换棋手算法不改变此局势估计。概率尚未独立验证。"
    if opening_uncovered:
        label = "开局训练依据不足，暂不显示百分比；不把算法强弱当作棋盘已经占优。"
    elif not available:
        label = "分析不足或缺少模型，暂不显示百分比"
    match = dict(
        status="estimated" if match_available else "unsupported",
        probabilities=(
            predict(black_score, board.empty_count, model, base.depth, disagreement, config)
            if match_available
            else None
        ),
        reference=sides,
        label=(
            "；".join(notes)
            if notes
            else "考虑当前双方算法和时限的预测；更换棋手后可能变化。尚未独立验证。"
        ),
    )
    return dict(
        status="estimated" if position_available else "provisional",
        probabilities=(
            predict(black_score, board.empty_count, model, base.depth, disagreement, neutral)
            if position_available
            else None
        ),
        score=black_score,
        depth=base.depth,
        nodes=base.nodes,
        exact=False,
        disagreement=disagreement,
        model_version="2.1",
        validation=None,
        reference=neutral,
        match_prediction=match,
        label=label,
    )


def analysis_worker(board, config, queue, cancel):
    """分析使用回合开始前的配置快照，不读取对弈进程中的可变状态。"""
    try:
        queue.put(analyze(board, cancel=cancel, config=config))
    except Exception as error:
        queue.put({"status": "error", "label": str(error), "probabilities": None})

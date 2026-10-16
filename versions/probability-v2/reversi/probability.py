"""概率 V2 的共同特征与推断；训练和线上分析必须使用同一特征定义。"""

import math

ALGORITHM_IDS = ("minimax", "alphabeta", "mcts", "hybrid", "hybrid_v2", "hybrid_v3")
# 前七项黑白互换不变；其余项必须取反，保证概率的颜色对称。
FEATURE_SIGNS = [1] * 7 + [-1] * 17
FEATURE_NAMES = [
    "bias",
    "phase",
    "phase_squared",
    "absolute_score",
    "depth_disagreement",
    "completed_depth",
    "mean_log_budget",
    "black_score",
    "score_phase",
    "score_phase_squared",
    "log_budget_difference",
    "budget_difference_early",
]
FEATURE_NAMES += ["algorithm_difference_" + name for name in ALGORITHM_IDS]
FEATURE_NAMES += ["early_algorithm_difference_" + name for name in ALGORITHM_IDS]


def softmax(logits):
    """先减最大值，避免指数溢出；三个类别始终归一化。"""
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def reference_config(config=None):
    """人类棋力未知：以与 AI 相同策略/时限作为参考，而非猜测人类实力。"""
    if config is None:
        config = {
            "mode": "eve",
            "black": {"algorithm": "hybrid_v3", "seconds": 0.2},
            "white": {"algorithm": "hybrid_v3", "seconds": 0.2},
        }
    sides = {color: dict(config[color]) for color in ("black", "white")}
    notes = []
    if config.get("mode") == "pve":
        human = "black" if config.get("human_color", 1) == 1 else "white"
        opponent = "white" if human == "black" else "black"
        sides[human] = dict(sides[opponent])
        notes.append("人类棋力未知，以与 AI 同策略、同时限作为参考")
    return sides, notes


def features(score, empty_count, depth=5, disagreement=0, config=None, bounds=(0.05, 0.5)):
    """融合局面、阶段、完成深度、层间分歧、双方算法和时长；不使用未来结果。"""
    sides, _ = reference_config(config)
    phase = max(0.0, min(1.0, (60 - empty_count) / 60))
    scaled = max(-6.0, min(6.0, score / 400))
    budgets = [
        math.log1p(20 * max(bounds[0], min(bounds[1], float(sides[c]["seconds"]))))
        for c in ("black", "white")
    ]
    delta = budgets[0] - budgets[1]
    differences = [
        int(sides["black"]["algorithm"] == name) - int(sides["white"]["algorithm"] == name)
        for name in ALGORITHM_IDS
    ]
    return [
        1,
        phase,
        phase * phase,
        abs(scaled),
        min(4.0, abs(disagreement) / 400),
        min(1.0, depth / 6),
        sum(budgets) / 2,
        scaled,
        scaled * phase,
        scaled * phase * phase,
        delta,
        delta * (1 - phase),
        *differences,
        *(value * (1 - phase) for value in differences),
    ]


def predict(score, empty_count, model, depth=5, disagreement=0, config=None):
    """输出未经独立验证的新模型估计，不挪用旧模型的误差指标。"""
    x = features(score, empty_count, depth, disagreement, config, tuple(model["budget_bounds"]))
    probabilities = softmax(
        [sum(a * b for a, b in zip(weights, x)) for weights in model["weights"]]
    )
    return dict(zip(("black", "draw", "white"), probabilities))

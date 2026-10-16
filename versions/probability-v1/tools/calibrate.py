"""生成自对弈数据并拟合胜/和/负概率；训练与验证按整盘棋划分。"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reversi.analysis import MODEL_PATH, features, softmax
from reversi.board import Board
from reversi.search import deterministic


def generate(games, seed):
    rng = random.Random(seed)
    records = []
    for game in range(games):
        board = Board()
        samples = []
        ply = 0
        # 多样化合法开局；之后双方均采用三层 Alpha-Beta。
        opening_length = rng.choice((4, 6, 8, 10, 12))
        while not board.terminal():
            moves = board.legal()
            if not moves:
                board = board.play(None)
                continue
            result = deterministic(board, 30, max_depth=3)
            if ply >= opening_length and ply % 2 == 0:
                samples.append(
                    {
                        "score": result.score * board.turn,
                        "empty": board.empty_count,
                        "black": str(board.black),
                        "white": str(board.white),
                        "turn": board.turn,
                    }
                )
            move = rng.choice(moves) if ply < opening_length else result.move
            board = board.play(move)
            ply += 1
        records.append({"game": game, "outcome": board.winner(), "samples": samples})
        if (game + 1) % 8 == 0:
            print("calibration games: %d/%d" % (game + 1, games), flush=True)
    return records


def dataset(records):
    examples = []
    for record in records:
        outcome = record["outcome"]
        for sample in record["samples"]:
            for sign in (1, -1):
                # 镜像样本保证黑白对称；仍保留在同一训练或验证分组。
                examples.append(
                    (features(sign * sample["score"], sample["empty"]), 1 - sign * outcome)
                )
    return examples


def fit(examples, epochs=450):
    weights = [[0.0] * 6 for _ in range(3)]
    first = [[0.0] * 6 for _ in range(3)]
    second = [[0.0] * 6 for _ in range(3)]
    for epoch in range(1, epochs + 1):
        gradient = [[0.0] * 6 for _ in range(3)]
        for x, label in examples:
            probabilities = softmax([sum(a * b for a, b in zip(w, x)) for w in weights])
            for c in range(3):
                residual = probabilities[c] - int(c == label)
                for j in range(6):
                    gradient[c][j] += residual * x[j]
        # 自行实现 Adam 与 L2 正则，避免依赖现成机器学习包。
        for c in range(3):
            for j in range(6):
                g = gradient[c][j] / len(examples) + (0.003 * weights[c][j] if j else 0)
                first[c][j] = 0.9 * first[c][j] + 0.1 * g
                second[c][j] = 0.999 * second[c][j] + 0.001 * g * g
                m = first[c][j] / (1 - 0.9**epoch)
                v = second[c][j] / (1 - 0.999**epoch)
                weights[c][j] -= 0.035 * m / (math.sqrt(v) + 1e-8)
    # 显式约束黑白镜像对称，消除优化器浮点累计造成的微小颜色偏差。
    signs = (1, -1, -1, 1, 1, 1)
    for j, sign in enumerate(signs):
        black = (weights[0][j] + sign * weights[2][j]) / 2
        weights[0][j], weights[2][j] = black, sign * black
        if sign == -1:
            weights[1][j] = 0.0
    return weights


def metrics(weights, examples):
    brier, loss = 0.0, 0.0
    bins = [{"count": 0, "predicted": 0.0, "observed": 0.0} for _ in range(10)]
    for x, label in examples:
        probabilities = softmax([sum(a * b for a, b in zip(w, x)) for w in weights])
        brier += sum((p - int(c == label)) ** 2 for c, p in enumerate(probabilities))
        loss -= math.log(max(1e-12, probabilities[label]))
        p = probabilities[0]
        bucket = bins[min(9, int(p * 10))]
        bucket["count"] += 1
        bucket["predicted"] += p
        bucket["observed"] += int(label == 0)
    for bucket in bins:
        if bucket["count"]:
            bucket["predicted"] /= bucket["count"]
            bucket["observed"] /= bucket["count"]
    return {
        "brier_multiclass": brier / len(examples),
        "log_loss": loss / len(examples),
        "samples_with_color_mirror": len(examples),
        "reliability_bins": bins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.games < 16:
        parser.error("至少需要 16 局，建议 128 局以上")
    path = MODEL_PATH.parent / "calibration_games.json"
    start = time.perf_counter()
    records = json.loads(path.read_text()) if args.reuse else generate(args.games, args.seed)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    order = list(records)
    random.Random(args.seed + 1).shuffle(order)
    split = int(len(order) * 0.75)
    train, validation = dataset(order[:split]), dataset(order[split:])
    weights = fit(train)
    validation_metrics = metrics(weights, validation)
    # 简单类别先验作为比较基线，不能只报告拟合模型的误差。
    frequencies = [sum(label == c for _, label in train) / len(train) for c in range(3)]
    prior_weights = [[math.log(max(p, 1e-6))] + [0.0] * 5 for p in frequencies]
    validation_metrics["prior_baseline_brier"] = metrics(prior_weights, validation)[
        "brier_multiclass"
    ]
    validation_metrics["train_games"] = split
    validation_metrics["heldout_games"] = len(order) - split
    model = {
        "version": 1,
        "reference": "三层 Alpha-Beta 对称自对弈；4~12 步随机合法开局；相同评估函数",
        "seed": args.seed,
        "weights": weights,
        "validation": validation_metrics,
        "limitations": "按整盘划分；相邻局面相关；仅代表参考策略，不能预测未知人类水平，也不是概率准确率。",
        "elapsed_seconds": time.perf_counter() - start,
    }
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

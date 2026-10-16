"""仅训练概率 V2：复用历史对局并补充非对称时限训练数据，不做准确性验证。"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reversi.analysis import MODEL_PATH, score_position
from reversi.board import Board
from reversi.experiments import opening
from reversi.probability import FEATURE_NAMES, FEATURE_SIGNS, features, softmax
from reversi.search import search

SOURCES = [
    "results/comparison_0.2s_0.5s.json",
    "results/evolution/screening_v2_v3_005.json",
    "results/evolution/validation_005.json",
    "results/evolution/validation_020.json",
    "results/evolution/validation_v1.json",
    "data/probability_v2_training_games.json",
]


def save(path, data):
    """训练中间产物原子保存，避免中断留下半份 JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def supplement():
    """同算法不同时限，交换颜色；用于学习时间差，不是保留验证集。"""
    path = ROOT / SOURCES[-1]
    records = (
        json.loads(path.read_text()) if path.exists() else {"games": [], "purpose": "training_only"}
    )
    schedule = [
        (algorithm, seed, reverse)
        for algorithm in ("alphabeta", "hybrid_v3", "minimax", "mcts")
        for seed in (905901, 905902)
        for reverse in (False, True)
    ]
    for index in range(len(records["games"]), len(schedule)):
        algorithm, seed, reverse = schedule[index]
        budgets = (0.2, 0.05) if reverse else (0.05, 0.2)
        initial = opening(seed, 6 if seed % 2 else 10)
        board, history = initial, []
        while not board.terminal():
            color = board.turn
            if not board.legal():
                history.append({"move": None, "color": color})
                board = board.play(None)
                continue
            seconds = budgets[0 if color == 1 else 1]
            result = search(board, algorithm, seconds, seed=seed + len(history))
            history.append(
                dict(result.as_dict(), color=color, algorithm=algorithm, seconds=seconds)
            )
            board = board.play(result.move)
        records["games"].append(
            dict(
                black=algorithm,
                white=algorithm,
                black_seconds=budgets[0],
                white_seconds=budgets[1],
                initial=dict(black=str(initial.black), white=str(initial.white), turn=initial.turn),
                history=history,
                winner=board.winner(),
                seed=seed,
            )
        )
        save(path, records)
        print(f"training games {index + 1}/{len(schedule)}", flush=True)


def collect():
    """稀疏抽取局面并用新版统一评分器重算，不直接混用旧分数。"""
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    dataset_path = ROOT / "data" / "probability_v2_samples.json"
    scorer_hash = hashlib.sha256(
        (ROOT / "reversi/advanced.py").read_bytes() + (ROOT / "reversi/analysis.py").read_bytes()
    ).hexdigest()
    data = {"version": 2, "source_hashes": hashes, "scorer_hash": scorer_hash, "games": []}
    if dataset_path.exists():
        previous = json.loads(dataset_path.read_text())
        if previous["source_hashes"] == hashes and previous["scorer_hash"] == scorer_hash:
            data = previous
    games = [
        dict(game, source=name)
        for name in SOURCES
        for game in json.loads((ROOT / name).read_text())["games"]
    ]
    for index in range(len(data["games"]), len(games)):
        game = games[index]
        start = game["initial"]
        board = Board(int(start["black"]), int(start["white"]), start["turn"])
        config = {
            "mode": "eve",
            "black": {
                "algorithm": game["black"],
                "seconds": game.get("black_seconds", game.get("seconds")),
            },
            "white": {
                "algorithm": game["white"],
                "seconds": game.get("white_seconds", game.get("seconds")),
            },
        }
        samples = []
        for ply, move in enumerate(game["history"]):
            if ply % 6 == 0 and not board.terminal():
                result, disagreement = score_position(board)
                if result.depth >= 3 and not result.exact:
                    samples.append(
                        {
                            "score": result.score * board.turn,
                            "empty": board.empty_count,
                            "depth": result.depth,
                            "disagreement": disagreement,
                            "black": str(board.black),
                            "white": str(board.white),
                            "turn": board.turn,
                        }
                    )
            board = board.play(move["move"])
        data["games"].append(
            {
                "source": game["source"],
                "index": index,
                "config": config,
                "winner": game["winner"],
                "samples": samples,
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(games):
            save(dataset_path, data)
            print(f"rescored training games {index + 1}/{len(games)}", flush=True)
    return data


def fit(data, epochs=260):
    """对称 Softmax＋L2＋Adam；每盘总权重相同，避免长棋谱重复支配拟合。"""
    examples = []
    bounds = [
        min(g["config"][c]["seconds"] for g in data["games"] for c in ("black", "white")),
        max(g["config"][c]["seconds"] for g in data["games"] for c in ("black", "white")),
    ]
    for game in data["games"]:
        for sample in game["samples"]:
            x = features(
                sample["score"],
                sample["empty"],
                sample["depth"],
                sample["disagreement"],
                game["config"],
                bounds,
            )
            mass = 1 / (2 * len(game["samples"]))
            examples.append((x, 1 - game["winner"], mass))
            examples.append(([a * b for a, b in zip(x, FEATURE_SIGNS)], 1 + game["winner"], mass))
    width = len(FEATURE_NAMES)
    weights = [[0.0] * width for _ in range(3)]
    first = [[0.0] * width for _ in range(3)]
    second = [[0.0] * width for _ in range(3)]
    total_mass = sum(mass for _, _, mass in examples)
    for epoch in range(1, epochs + 1):
        gradient = [[0.0] * width for _ in range(3)]
        for x, label, mass in examples:
            probabilities = softmax([sum(a * b for a, b in zip(w, x)) for w in weights])
            for c in range(3):
                residual = (probabilities[c] - int(c == label)) * mass
                for j in range(width):
                    gradient[c][j] += residual * x[j]
        for c in range(3):
            for j in range(width):
                g = gradient[c][j] / total_mass + (0.01 * weights[c][j] if j else 0)
                first[c][j] = 0.9 * first[c][j] + 0.1 * g
                second[c][j] = 0.999 * second[c][j] + 0.001 * g * g
                m = first[c][j] / (1 - 0.9**epoch)
                v = second[c][j] / (1 - 0.999**epoch)
                weights[c][j] -= 0.035 * m / (math.sqrt(v) + 1e-8)
        if epoch % 65 == 0:
            print(f"fit epoch {epoch}/{epochs}", flush=True)
    for j, sign in enumerate(FEATURE_SIGNS):
        value = (weights[0][j] + sign * weights[2][j]) / 2
        weights[0][j], weights[2][j] = value, sign * value
        if sign == -1:
            weights[1][j] = 0
    pairs = set()
    for game in data["games"]:
        a, b = game["config"]["black"]["algorithm"], game["config"]["white"]["algorithm"]
        pairs.update((a + "|" + b, b + "|" + a))
    return {
        "version": 2,
        "name": "V3五层分析＋对局条件化概率",
        "weights": weights,
        "feature_names": FEATURE_NAMES,
        "feature_signs": FEATURE_SIGNS,
        "budget_bounds": bounds,
        "training": {
            "games": len(data["games"]),
            "samples_with_mirror": len(examples),
            "algorithm_pairs": sorted(pairs),
            "source_hashes": data["source_hashes"],
            "scorer_hash": data["scorer_hash"],
            "epochs": epochs,
            "l2": 0.01,
            "game_weighting": "each game has equal total weight",
            "all_data_used_for_training": True,
        },
        "validation": None,
        "validation_status": "not_performed_at_user_request",
        "limitations": "未进行独立准确性验证；训练开局和算法覆盖有限。人类以同对手棋力参考，超范围时长按边界输入，不保证实际胜率精确。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="复用已生成的新版训练特征，仅重新拟合")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.reuse:
        data = json.loads((ROOT / "data" / "probability_v2_samples.json").read_text())
    else:
        supplement()
        data = collect()
    model = fit(data)
    model["training"]["elapsed_seconds"] = time.perf_counter() - started
    save(MODEL_PATH, model)
    print(f"saved {MODEL_PATH}; training only, no accuracy validation", flush=True)


if __name__ == "__main__":
    main()

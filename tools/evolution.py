"""冻结候选后运行独立验证；按完整换色组统计，避免把两局误当独立样本。"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reversi.experiments import opening, play_game, summarize
from reversi.search import ALGORITHMS


def paired_interval(scores):
    """对开局组均分做配对 bootstrap，给出描述性的 95% 区间。"""
    rng = random.Random(294518)
    means = sorted(sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(10000))
    return [means[250], means[9749]]


def comparisons(games, candidate):
    groups = {}
    for game in games:
        color = 1 if game["black"] == candidate else -1
        opponent = game["white"] if color == 1 else game["black"]
        row = groups.setdefault(
            (game["seconds"], opponent),
            {
                "scores": [],
                "pairs": {},
                "times": [],
                "depths": [],
                "opponent_depths": [],
                "max_overrun": 0,
            },
        )
        score = 0.5 if game["winner"] == 0 else float(game["winner"] == color)
        row["scores"].append(score)
        row["pairs"].setdefault(game["opening_seed"], []).append(score)
        for move in game["history"]:
            if move["move"] is None:
                continue
            if move["color"] == color:
                row["times"].append(move["elapsed"])
                row["depths"].append(move["depth"])
                row["max_overrun"] = max(row["max_overrun"], move["elapsed"] - game["seconds"])
            else:
                row["opponent_depths"].append(move["depth"])
    rows = []
    for (seconds, opponent), row in groups.items():
        scores = row["scores"]
        pair_scores = [sum(pair) / 2 for pair in row["pairs"].values() if len(pair) == 2]
        rows.append(
            dict(
                candidate=candidate,
                opponent=opponent,
                seconds=seconds,
                games=len(scores),
                wins=scores.count(1),
                draws=scores.count(0.5),
                losses=scores.count(0),
                win_rate=scores.count(1) / len(scores),
                score_rate=sum(scores) / len(scores),
                paired_bootstrap_95=paired_interval(pair_scores) if len(pair_scores) >= 2 else None,
                completed_pairs=len(pair_scores),
                mean_seconds=sum(row["times"]) / len(row["times"]),
                max_overrun=row["max_overrun"],
                mean_depth=sum(row["depths"]) / len(row["depths"]),
                opponent_mean_depth=sum(row["opponent_depths"]) / len(row["opponent_depths"]),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=list(ALGORITHMS), default="hybrid_v3")
    parser.add_argument(
        "--opponents", nargs="+", choices=list(ALGORITHMS), default=["alphabeta", "minimax", "mcts"]
    )
    parser.add_argument("--seconds", nargs="+", type=float, default=[0.05])
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=902610)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.candidate in args.opponents
        or args.pairs < 2
        or any(not 0 < s <= 59.5 for s in args.seconds)
    ):
        parser.error("候选不能作为自己的对手；至少两组换色；时间范围 (0, 59.5]")
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        candidate=args.candidate,
        config=vars(args),
        status="running",
        games=[],
        comparisons=[],
        environment=dict(python=platform.python_version(), platform=platform.platform()),
        protocol="候选参数冻结；未参与筛选的新种子；6/8/10/12步随机开局循环；每开局换色；串行同上限；无旁路分析/跨局缓存/开局库",
        hashes={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ("reversi/search.py", "reversi/advanced.py", "reversi/board.py")
        },
    )
    start = time.perf_counter()
    total = len(args.seconds) * len(args.opponents) * args.pairs * 2
    for seconds in args.seconds:
        for opponent in args.opponents:
            for pair in range(args.pairs):
                seed = args.seed + pair
                plies = (6, 8, 10, 12)[pair % 4]
                initial = opening(seed, plies)
                for black, white in ((args.candidate, opponent), (opponent, args.candidate)):
                    game = play_game(black, white, seconds, initial, seed * 100)
                    game.update(opening_seed=seed, opening_plies=plies)
                    result["games"].append(game)
                    result["comparisons"] = comparisons(result["games"], args.candidate)
                    result["summary"] = summarize(result["games"])
                    result["elapsed"] = time.perf_counter() - start
                    result["completed"], result["total"] = len(result["games"]), total
                    result["status"] = "completed" if len(result["games"]) == total else "running"
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    temporary.replace(path)
                    last = result["comparisons"][-1]
                    print(
                        f"{len(result['games'])}/{total} {seconds}s vs {opponent}: {last['wins']}/{last['draws']}/{last['losses']}",
                        flush=True,
                    )
    print(json.dumps(result["comparisons"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

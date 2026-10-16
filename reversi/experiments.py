"""可复现的成对换色实验：统一时间上限，完整记录配置、棋谱与统计。"""

import itertools
import platform
import random
import time

from .board import Board, coordinate, count
from .search import ALGORITHMS, search


def opening(seed: int, plies=6):
    """给定种子生成合法开局，供两场换色对局和不同时间档位复用。"""
    rng = random.Random(seed)
    board = Board()
    for _ in range(plies):
        moves = board.legal()
        if board.terminal():
            break
        board = board.play(rng.choice(moves) if moves else None)
    return board


def play_game(
    black: str, white: str, seconds: float, initial: Board, seed: int, cancel=None, progress=None
):
    board = initial
    history = []
    decision = 0
    while not board.terminal():
        if cancel is not None and cancel.is_set():
            return None
        color = board.turn
        algorithm = black if color == 1 else white
        moves = board.legal()
        if not moves:
            history.append(
                {
                    "color": color,
                    "move": None,
                    "coordinate": "PASS",
                    "algorithm": algorithm,
                    "elapsed": 0,
                    "nodes": 0,
                    "iterations": 0,
                    "depth": 0,
                }
            )
            board = board.play(None)
            continue
        result = search(board, algorithm, seconds, cancel=cancel, seed=seed + decision)
        if cancel is not None and cancel.is_set():
            return None
        history.append(
            dict(
                result.as_dict(),
                color=color,
                coordinate=coordinate(result.move),
                algorithm=algorithm,
            )
        )
        board = board.play(result.move)
        decision += 1
        if progress:
            progress(
                {
                    "ply": decision,
                    "black_count": count(board.black),
                    "white_count": count(board.white),
                }
            )
    return {
        "black": black,
        "white": white,
        "seconds": seconds,
        "seed": seed,
        "initial": {"black": str(initial.black), "white": str(initial.white), "turn": initial.turn},
        "winner": board.winner(),
        "black_count": count(board.black),
        "white_count": count(board.white),
        "history": history,
    }


def summarize(games):
    """按算法与时间档位聚合完整棋局，胜率与含和棋半分的得分率分开。"""
    rows = {}
    for game in games:
        for color, algorithm in ((1, game["black"]), (-1, game["white"])):
            key = (game["seconds"], algorithm)
            row = rows.setdefault(
                key,
                {
                    "algorithm": algorithm,
                    "seconds": game["seconds"],
                    "games": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "moves": 0,
                    "time": 0.0,
                    "nodes": 0,
                    "iterations": 0,
                    "depth_total": 0,
                },
            )
            row["games"] += 1
            row[
                "wins" if game["winner"] == color else "draws" if not game["winner"] else "losses"
            ] += 1
            for move in game["history"]:
                if move["color"] == color and move["move"] is not None:
                    row["moves"] += 1
                    row["time"] += move["elapsed"]
                    row["nodes"] += move["nodes"]
                    row["iterations"] += move["iterations"]
                    row["depth_total"] += move["depth"]
    for row in rows.values():
        row["win_rate"] = row["wins"] / row["games"]
        row["score_rate"] = (row["wins"] + row["draws"] / 2) / row["games"]
        row["mean_seconds"] = row["time"] / max(1, row["moves"])
        row["mean_depth"] = row["depth_total"] / max(1, row["moves"])
    return sorted(
        rows.values(), key=lambda row: (row["seconds"], -row["score_rate"], row["algorithm"])
    )


def run(config, cancel=None, progress=None):
    """串行执行时间档位 × 算法配对 × 开局组 × 颜色交换四重实验。"""
    algorithms = config["algorithms"]
    budgets = config["budgets"]
    pairs = config["pairs"]
    seed = config.get("seed", 20261014)
    total = len(list(itertools.combinations(algorithms, 2))) * len(budgets) * pairs * 2
    results = {
        "config": config,
        "protocol": "同一合法开局成对换色；同档位上限；单搜索进程串行；无胜率分析",
        "environment": {"python": platform.python_version(), "system": platform.platform()},
        "games": [],
        "summary": [],
        "total": total,
        "completed": 0,
        "status": "running",
    }
    start = time.perf_counter()
    for seconds in budgets:
        for a, b in itertools.combinations(algorithms, 2):
            for pair in range(pairs):
                initial = opening(seed + pair)
                for black, white in ((a, b), (b, a)):

                    def on_move(data):
                        if progress:
                            progress(
                                {
                                    "type": "move",
                                    "data": dict(
                                        data,
                                        black=black,
                                        white=white,
                                        seconds=seconds,
                                        completed=results["completed"],
                                        total=total,
                                    ),
                                }
                            )

                    game = play_game(
                        black, white, seconds, initial, seed + pair * 100, cancel, on_move
                    )
                    if game is None:
                        results["status"] = "cancelled"
                        results["elapsed"] = time.perf_counter() - start
                        return results
                    results["games"].append(game)
                    results["completed"] += 1
                    results["summary"] = summarize(results["games"])
                    if progress:
                        progress({"type": "snapshot", "data": results})
    results["elapsed"] = time.perf_counter() - start
    results["status"] = "completed"
    return results


def experiment_worker(config, queue, cancel):
    try:
        result = run(config, cancel, lambda data: queue.put(data))
        queue.put({"type": "done", "data": result})
    except Exception as error:
        queue.put({"type": "error", "message": str(error)})

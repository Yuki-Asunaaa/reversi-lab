"""无需打开界面的算法实验入口；支持更短时间用于快速回归。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reversi.experiments import run
from reversi.search import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS)
    )
    parser.add_argument("--seconds", nargs="+", type=float, default=[0.2])
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    if (
        len(set(args.algorithms)) < 2
        or not all(0 < n <= 59.5 for n in args.seconds)
        or args.pairs < 1
    ):
        parser.error("至少两种不同算法、正数时间上限（不超过59.5秒）和一组换色对局")
    config = {
        "algorithms": list(dict.fromkeys(args.algorithms)),
        "budgets": args.seconds,
        "pairs": args.pairs,
        "seed": args.seed,
    }

    def progress(data):
        if data["type"] == "snapshot":
            print(
                "completed %d/%d" % (data["data"]["completed"], data["data"]["total"]), flush=True
            )

    result = run(config, progress=progress)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

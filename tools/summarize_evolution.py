"""把已完成的独立验证汇总到界面、CSV 和进化日志；不会把中止数据冒充完成。"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results" / "evolution"
FILES = ["validation_005.json", "validation_020.json", "validation_v1.json"]
NAMES = {
    "hybrid_v3": "V3",
    "hybrid": "V1",
    "alphabeta": "Alpha-Beta",
    "minimax": "Minimax",
    "mcts": "MCTS",
}


def main():
    rows, sources = [], []
    for name in FILES:
        path = DIRECTORY / name
        if not path.exists():
            continue
        experiment = json.loads(path.read_text(encoding="utf-8"))
        if experiment["status"] != "completed":
            continue
        rows.extend(experiment["comparisons"])
        sources.append(
            dict(
                file=name,
                games=experiment["completed"],
                hashes=experiment["hashes"],
                environment=experiment["environment"],
            )
        )
    short = [
        r
        for r in rows
        if r["seconds"] == 0.05 and r["opponent"] in ("alphabeta", "minimax", "mcts")
    ]
    passed = len(short) == 3 and all(
        r["score_rate"] >= 0.65 and r["paired_bootstrap_95"][0] > 0.5 for r in short
    )
    conclusion = (
        "0.05 秒独立验证中，V3 对三种基线均取得明显优势：各组得分率超过 65%，配对得分率 95% 区间下界均高于 50%。结论限定于本机与所测开局；不宣称无限时间优势。"
        if passed
        else "独立验证结果如下；不根据少量对局保证普遍优势。"
    )
    # 全胜样本的普通 bootstrap 会退化成 [1, 1]，不能解读为确定胜率。
    for row in rows:
        interval = row["paired_bootstrap_95"]
        row["interval_note"] = ""
        if interval and interval[0] == interval[1]:
            row["paired_bootstrap_95"] = None
            row["interval_note"] = "样本无组间差异，bootstrap 区间退化，不代表真实胜率确定"
    summary = dict(
        conclusion=conclusion,
        criterion="每个基线同档位得分率≥65%，按开局组 bootstrap 的95%区间下界>50%；描述性实测门槛，非普遍棋力保证",
        comparisons=rows,
        sources=sources,
    )
    (DIRECTORY / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (DIRECTORY / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "## 独立验证实测结果",
        "",
        conclusion,
        "",
        "| 对手 | 每步上限 | 局数 | V3 胜 / 和 / 负 | 胜率 | 得分率 | 配对得分率 95% 区间 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        interval = (
            "～".join(f"{p:.1%}" for p in row["paired_bootstrap_95"])
            if row["paired_bootstrap_95"]
            else "样本区间退化，不作确定性推断"
        )
        lines.append(
            f"| {NAMES[row['opponent']]} | {row['seconds']}s | {row['games']} | {row['wins']} / {row['draws']} / {row['losses']} | {row['win_rate']:.1%} | {row['score_rate']:.1%} | {interval} |"
        )
    lines += [
        "",
        "### 计时与搜索深度",
        "",
        "| 对手 | 每步上限 | V3 平均搜索耗时 | V3 最大越界 | V3 / 对手平均完成深度 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {NAMES[row['opponent']]} | {row['seconds']}s | {row['mean_seconds']:.4f}s | {row['max_overrun'] * 1000:.3f}ms | {row['mean_depth']:.2f} / {row['opponent_mean_depth']:.2f} |"
        )
    lines += [
        "",
        "深度来自各自实际对局轨迹，不能据此单独归因于某项优化；MCTS 不采用固定深度，表中 0 不代表没有探索。",
        "本次未做所有组件的单独消融，所以提升属于整个 V3 组合，不能量化每个特征分别贡献多少。",
        "V1 对照只有 6 个换色组，全部获胜使普通 bootstrap 退化为 [1, 1]；汇总故意不显示该区间，12/12 并不保证今后必胜。原始文件保留原计算结果。",
        "",
        "数据文件：" + "、".join(f"`results/evolution/{s['file']}`" for s in sources) + "。",
        "汇总文件：`results/evolution/summary.json`、`summary.csv`；运行 `python3 tools/summarize_evolution.py` 可从完整数据重建。",
    ]
    document = ROOT / "EVOLUTION.md"
    content = document.read_text(encoding="utf-8").split("\n<!-- VERIFIED_RESULTS -->")[0]
    document.write_text(
        content + "\n<!-- VERIFIED_RESULTS -->\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(conclusion)
    print(f'{len(rows)} comparisons, {sum(row["games"] for row in rows)} games')


if __name__ == "__main__":
    main()

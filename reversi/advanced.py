"""创新策略迭代：V2 混合搜索机制；V3 再混合动态局面特征。

均为本项目自行实现的经典技术组合，不把 PVS 或置换表宣称为原创算法。
局面缓存只在一次落子内生效，不跨对局学习，也不偷用对手思考时间。
"""

import math
from .board import Board, CORNERS, FULL, WEIGHTS, count, indices, stable_edges
from .search import Context, Result, SearchStopped

INNER = 0x7E7E7E7E7E7E7E7E
# 相同权重一次合并为位掩码，叶节点无需逐棋子遍历。
WEIGHT_MASKS = tuple(
    (w, sum(1 << i for i, value in enumerate(WEIGHTS) if value == w)) for w in sorted(set(WEIGHTS))
)
CORNER_NEIGHBORS = ((0, (1, 8), 9), (7, (6, 15), 14), (56, (48, 57), 49), (63, (55, 62), 54))


def legal(own, other):
    """批量扩展棋链；横向与斜向屏蔽 A/H 列，避免跨行连通。"""
    empty = FULL ^ (own | other)
    result = 0
    for step in (1, 7, 8, 9):
        mask = other if step == 8 else other & INNER
        ray = mask & (own << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        result |= ray << step
        ray = mask & (own >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        result |= ray >> step
    return result & empty


def play(own, other, move):
    """内部搜索只接收已验证合法着法，返回换手后的 (己方, 对方)。"""
    bit = 1 << move
    flipped = 0
    for step in (1, 7, 8, 9):
        mask = other if step == 8 else other & INNER
        ray = mask & (bit << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        ray |= mask & (ray << step)
        if own & (ray << step):
            flipped |= ray
        ray = mask & (bit >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        ray |= mask & (ray >> step)
        if own & (ray >> step):
            flipped |= ray
    return other ^ flipped, own | flipped | bit


def neighbors(bits):
    """周围八格的并集，用于前沿子和潜在行动力。"""
    left = (bits & 0xFEFEFEFEFEFEFEFE) >> 1
    right = (bits & 0x7F7F7F7F7F7F7F7F) << 1
    row = bits | left | right
    return (left | right | (row << 8) | (row >> 8)) & FULL


def evaluation(own, other, own_moves, other_moves, empty_count, version):
    """V2 与基线评估等价；V3 用角点状态、潜在行动力修正静态位置价值。"""
    difference = count(own) - count(other)
    if not own_moves and not other_moves:
        return (
            100000 + difference if difference > 0 else -100000 + difference if difference < 0 else 0
        )
    empty = FULL ^ (own | other)
    frontier = neighbors(empty)
    phase = (60 - empty_count) / 60
    positional = sum(
        weight * (count(own & mask) - count(other & mask)) for weight, mask in WEIGHT_MASKS
    )
    mobility = count(own_moves) - count(other_moves)
    corners = count(own & CORNERS) - count(other & CORNERS)
    stable = count(stable_edges(own)) - count(stable_edges(other))
    exposed = count(other & frontier) - count(own & frontier)
    if version == 2:
        return int(
            positional * (1 - phase * 0.5)
            + mobility * (22 - 12 * phase)
            + corners * 160
            + stable * 18
            + exposed * (9 - 6 * phase)
            + difference * (1 + 18 * phase**4)
        )
    # 已经占角后，角旁的 X/C 格不再一律危险；空角旁则继续惩罚。
    correction = 0
    for corner, c_squares, x_square in CORNER_NEIGHBORS:
        if (own | other) & (1 << corner):
            for square in (*c_squares, x_square):
                sign = int(bool(own & (1 << square))) - int(bool(other & (1 << square)))
                correction += sign * (6 - WEIGHTS[square])
    potential = count(empty & neighbors(other)) - count(empty & neighbors(own))
    return int(
        (positional + correction) * 0.5
        + mobility * (30 - 12 * phase)
        + corners * 220
        + stable * 28
        + exposed * (14 - 8 * phase)
        + potential * (7 - 5 * phase)
        + difference * (-2 + 24 * phase**4)
    )


def advanced(board: Board, seconds, version=2, cancel=None, progress=None, max_depth=None):
    """PVS + 迭代加深 + 有界置换表 + 历史排序；整层完成才替换落子。

    PVS 首分支用完整窗口，后续分支先用零窗口验证，可能更好时重搜。
    缓存明确保存精确值/下界/上界；不把剪枝返回的界限误当精确值。
    """
    context = Context(seconds, cancel, progress)
    own, other = board.own_other
    root_moves = indices(legal(own, other))
    empty_count = board.empty_count
    strategy = f"V{version} · PVS＋置换表" + ("＋动态评估" if version == 3 else "＋基线评估")
    result = Result(move=root_moves[0] if root_moves else None, strategy=strategy)
    table = {}
    history = [[0] * 64 for _ in range(2)]
    killers = {}
    infinity = 200000

    def visit(us, them, depth, alpha, beta, empties, ply):
        context.nodes += 1
        context.check()
        key = (us, them)
        entry = table.get(key)
        original_alpha, original_beta = alpha, beta
        preferred = entry[3] if entry else None
        if entry and entry[0] >= depth:
            _, value, flag, _ = entry
            if flag == 0:
                return value
            if flag == 1:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        moves = legal(us, them)
        if not moves:
            opponent_moves = legal(them, us)
            if not opponent_moves:
                return evaluation(us, them, moves, opponent_moves, empties, version)
            # 强制跳过不消耗落子深度，且必须先于深度为零判断。
            return -visit(them, us, depth, -beta, -alpha, empties, ply + 1)
        if depth == 0:
            return evaluation(us, them, moves, legal(them, us), empties, version)
        options = indices(moves)
        killer = killers.get(ply)
        side_history = history[ply & 1]
        options.sort(
            key=lambda m: (
                m == preferred,
                bool((1 << m) & CORNERS),
                m == killer,
                side_history[m] + WEIGHTS[m] * 4,
            ),
            reverse=True,
        )
        best, best_move = -infinity, options[0]
        for index, move in enumerate(options):
            child_us, child_them = play(us, them, move)
            if index == 0:
                value = -visit(child_us, child_them, depth - 1, -beta, -alpha, empties - 1, ply + 1)
            else:
                value = -visit(
                    child_us, child_them, depth - 1, -alpha - 1, -alpha, empties - 1, ply + 1
                )
                if alpha < value < beta:
                    value = -visit(
                        child_us, child_them, depth - 1, -beta, -alpha, empties - 1, ply + 1
                    )
            if value > best:
                best, best_move = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                killers[ply] = move
                side_history[move] += depth * depth
                break
        # 满表后停止增加新键，避免一次巨大清空导致时限抖动。
        flag = 2 if best <= original_alpha else 1 if best >= original_beta else 0
        if len(table) < 180000 or key in table:
            table[key] = (depth, best, flag, best_move)
        return best

    limit = min(empty_count, empty_count if max_depth is None else max_depth)
    try:
        for depth in range(1, limit + 1):
            # V3 使用上一层分数建立愿望窗口；失败必须扩大到完整窗口重搜。
            if version == 3 and depth >= 4 and abs(result.score) < 90000:
                lo, hi = result.score - 90, result.score + 90
                value = visit(own, other, depth, lo, hi, empty_count, 0)
                if value <= lo or value >= hi:
                    value = visit(own, other, depth, -infinity, infinity, empty_count, 0)
            else:
                value = visit(own, other, depth, -infinity, infinity, empty_count, 0)
            entry = table.get((own, other))
            result.move = entry[3] if root_moves and entry else None
            result.score, result.depth = value, depth
            result.exact = depth >= empty_count or abs(value) >= 100000
            result.candidates = [{"move": result.move, "score": value}]
            context.emit(result, force=True)
            if result.exact:
                break
    except SearchStopped:
        pass
    context.emit(result, force=True)
    return result

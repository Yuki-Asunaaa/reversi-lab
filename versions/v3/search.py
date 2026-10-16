"""四种自行实现的搜索策略；统一接口、计时口径和进度回调。"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .board import Board, CORNERS, WEIGHTS, count, evaluate, indices, legal_bits

ALGORITHMS = {
    "minimax": "Minimax",
    "alphabeta": "Alpha-Beta",
    "mcts": "MCTS",
    "hybrid": "V1 · 阶段混合（原版）",
    "hybrid_v2": "V2 · 缓存主变化搜索",
    "hybrid_v3": "V3 · 动态评估混合",
}
ENDGAME_EMPTY = 12


class SearchStopped(Exception):
    """时间到或外部取消时中断当前层，保留上一完整层的结果。"""


@dataclass
class Result:
    move: Optional[int] = None
    score: float = 0
    depth: int = 0
    nodes: int = 0
    iterations: int = 0
    elapsed: float = 0
    exact: bool = False
    strategy: str = ""
    candidates: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class Context:
    def __init__(self, seconds: float, cancel=None, progress: Optional[Callable] = None):
        self.start = time.perf_counter()
        self.deadline = self.start + max(0.001, seconds)
        self.cancel = cancel
        self.progress = progress
        self.nodes = 0
        self.last_emit = 0.0

    def check(self):
        if time.perf_counter() >= self.deadline or (
            self.cancel is not None and self.cancel.is_set()
        ):
            raise SearchStopped()

    def emit(self, result: Result, force=False):
        now = time.perf_counter()
        result.elapsed = now - self.start
        result.nodes = self.nodes
        if self.progress and (force or now - self.last_emit > 0.15):
            self.progress(result.as_dict())
            self.last_emit = now


def ordered(board: Board, moves: List[int], preferred=None) -> List[int]:
    """共同使用角点、位置权重排序；Alpha-Beta 的差异主要来自剪枝。"""
    return sorted(moves, key=lambda move: (move == preferred, WEIGHTS[move]), reverse=True)


def deterministic(
    board: Board,
    seconds: float,
    pruning: bool = True,
    cancel=None,
    progress=None,
    max_depth: Optional[int] = None,
) -> Result:
    """迭代加深 Negamax；pruning=False 为不剪枝的完整 Minimax。"""
    context = Context(seconds, cancel, progress)
    moves = board.legal()
    result = Result(
        move=moves[0] if moves else None,
        score=evaluate(board),
        strategy="Alpha-Beta" if pruning else "Minimax",
    )
    if board.terminal():
        result.exact = True
        return result

    def visit(position: Board, depth: int, alpha: float, beta: float) -> int:
        context.nodes += 1
        context.check()
        options = position.legal()
        if not options:
            if position.terminal():
                return evaluate(position)
            # PASS 不填入格子，因此不消耗搜索深度。
            return -visit(position.play(None), depth, -beta, -alpha)
        if depth == 0:
            return evaluate(position)
        best = -math.inf
        for move in ordered(position, options):
            score = -visit(position.play(move), depth - 1, -beta, -alpha)
            best = max(best, score)
            if pruning:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        return int(best)

    limit = min(board.empty_count, max_depth if max_depth is not None else board.empty_count)
    try:
        for depth in range(1, limit + 1):
            context.check()
            alpha = -math.inf
            candidates = []
            if not moves:
                score = -visit(board.play(None), depth, -math.inf, math.inf)
                candidates.append({"move": None, "score": score})
            else:
                for move in ordered(board, moves, result.move):
                    # 共享根节点 alpha；未胜过当前最佳者的分数可能是界限。
                    score = -visit(
                        board.play(move), depth - 1, -math.inf, -alpha if pruning else math.inf
                    )
                    candidates.append({"move": move, "score": score})
                    alpha = max(alpha, score)
            best = max(candidates, key=lambda candidate: candidate["score"])
            result.move, result.score, result.depth = best["move"], best["score"], depth
            result.candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
            result.exact = depth >= board.empty_count
            context.emit(result, force=True)
            if result.exact:
                break
    except SearchStopped:
        pass
    context.emit(result, force=True)
    return result


class Node:
    """节点保存黑方期望得分；选择时转换为当前行棋方的视角。"""

    __slots__ = (
        "board",
        "move",
        "parent",
        "children",
        "untried",
        "visits",
        "black_score",
        "terminal",
    )

    def __init__(self, board: Board, move=None, parent=None):
        self.board, self.move, self.parent = board, move, parent
        self.children = []
        self.terminal = board.terminal()
        self.untried = board.legal() if not self.terminal else []
        if not self.untried and not self.terminal:
            self.untried = [None]
        self.visits = 0
        self.black_score = 0.0


def rollout_move(board: Board, rng: random.Random) -> Optional[int]:
    """模拟策略：优先角点，尽量避开空角旁危险格，其余随机探索。"""
    moves = board.legal()
    if not moves:
        return None
    corners = [move for move in moves if (1 << move) & CORNERS]
    if corners:
        return rng.choice(corners)
    dangerous = set()
    occupied = board.black | board.white
    for corner, adjacent in (
        (0, (1, 8, 9)),
        (7, (6, 14, 15)),
        (56, (48, 49, 57)),
        (63, (54, 55, 62)),
    ):
        if not occupied & (1 << corner):
            dangerous.update(adjacent)
    safe = [move for move in moves if move not in dangerous]
    return rng.choice(safe or moves)


def monte_carlo(board: Board, seconds: float, cancel=None, progress=None, seed=0) -> Result:
    """完整 UCT：选择、扩展、模拟到终局、回传；和棋按 0.5 得分。"""
    context = Context(seconds, cancel, progress)
    rng = random.Random(seed)
    root = Node(board)
    moves = board.legal()
    result = Result(move=moves[0] if moves else None, strategy="MCTS")

    def update_result():
        if root.children:
            best = max(root.children, key=lambda child: child.visits)
            result.move = best.move
            q = best.black_score / best.visits if best.visits else 0.5
            result.score = q if board.turn == 1 else 1 - q
            result.candidates = sorted(
                [
                    {
                        "move": child.move,
                        "visits": child.visits,
                        "expected_score": (
                            (
                                child.black_score / child.visits
                                if board.turn == 1
                                else 1 - child.black_score / child.visits
                            )
                            if child.visits
                            else 0.5
                        ),
                    }
                    for child in root.children
                ],
                key=lambda item: item["visits"],
                reverse=True,
            )
        result.iterations = root.visits
        context.emit(result)

    if root.terminal:
        result.exact = True
        return result
    try:
        while True:
            context.check()
            node = root
            # UCT 的探索项让访问较少的分支保留被搜索的机会。
            while not node.untried and node.children:
                log_parent = math.log(max(1, node.visits))

                def uct(child):
                    if not child.visits:
                        return math.inf
                    black_q = child.black_score / child.visits
                    q = black_q if node.board.turn == 1 else 1 - black_q
                    return q + math.sqrt(2 * log_parent / child.visits)

                node = max(node.children, key=uct)
                context.nodes += 1
                context.check()
            if node.untried:
                move = node.untried.pop(rng.randrange(len(node.untried)))
                child = Node(node.board.play(move), move, node)
                node.children.append(child)
                node = child
                context.nodes += 1
            position = node.board
            while not position.terminal():
                context.check()
                position = position.play(rollout_move(position, rng))
                context.nodes += 1
            black_score = (position.winner() + 1) / 2
            while node is not None:
                node.visits += 1
                node.black_score += black_score
                node = node.parent
            update_result()
    except SearchStopped:
        pass
    update_result()
    context.emit(result, force=True)
    return result


def search(
    board: Board, algorithm: str, seconds: float, cancel=None, progress=None, seed=0
) -> Result:
    """统一入口；混合模型只改变阶段策略，不偷用额外的思考预算。"""
    if algorithm not in ALGORITHMS:
        raise ValueError("未知算法")
    seconds = min(59.5, max(0.001, seconds))
    if algorithm == "minimax":
        return deterministic(board, seconds, False, cancel, progress)
    if algorithm == "alphabeta":
        return deterministic(board, seconds, True, cancel, progress)
    if algorithm == "mcts":
        return monte_carlo(board, seconds, cancel, progress, seed)
    if algorithm in ("hybrid_v2", "hybrid_v3"):
        from .advanced import advanced

        return advanced(board, seconds, int(algorithm[-1]), cancel, progress)
    strategy = (
        "混合 · 残局 Alpha-Beta" if board.empty_count <= ENDGAME_EMPTY else "混合 · 中前盘 MCTS"
    )

    def hybrid_progress(data):
        data["strategy"] = strategy
        if progress:
            progress(data)

    if board.empty_count <= ENDGAME_EMPTY:
        result = deterministic(board, seconds, True, cancel, hybrid_progress)
    else:
        result = monte_carlo(board, seconds, cancel, hybrid_progress, seed)
    result.strategy = strategy
    return result


def worker(board: Board, algorithm: str, seconds: float, queue, cancel, seed=0):
    """进程边界：搜索不会阻塞网页服务、动画和计时器。"""
    try:
        result = search(
            board,
            algorithm,
            seconds,
            cancel,
            lambda data: queue.put({"type": "progress", "data": data}),
            seed,
        )
        queue.put({"type": "done", "data": result.as_dict()})
    except Exception as error:
        queue.put({"type": "error", "message": str(error)})

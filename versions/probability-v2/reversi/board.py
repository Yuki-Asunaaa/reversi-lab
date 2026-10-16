"""规则引擎：用两个 64 位整数保存棋盘，最低位对应左上角 A1。"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

FULL = (1 << 64) - 1
A_FILE = 0x0101010101010101
H_FILE = 0x8080808080808080
CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
EDGES = 0xFF818181818181FF
WEIGHTS = (
    100,
    -24,
    12,
    6,
    6,
    12,
    -24,
    100,
    -24,
    -48,
    -4,
    -3,
    -3,
    -4,
    -48,
    -24,
    12,
    -4,
    8,
    2,
    2,
    8,
    -4,
    12,
    6,
    -3,
    2,
    1,
    1,
    2,
    -3,
    6,
    6,
    -3,
    2,
    1,
    1,
    2,
    -3,
    6,
    12,
    -4,
    8,
    2,
    2,
    8,
    -4,
    12,
    -24,
    -48,
    -4,
    -3,
    -3,
    -4,
    -48,
    -24,
    100,
    -24,
    12,
    6,
    6,
    12,
    -24,
    100,
)


def count(bits: int) -> int:
    """兼容 Python 3.9 与更新版本的位计数。"""
    return bits.bit_count() if hasattr(int, "bit_count") else bin(bits).count("1")


def indices(bits: int) -> List[int]:
    """逐个取最低有效位，避免遍历全部 64 个格子。"""
    result = []
    while bits:
        low = bits & -bits
        result.append(low.bit_length() - 1)
        bits ^= low
    return result


def shift(bits: int, direction: int) -> int:
    """八方向移位；列掩码阻止 H 列向下一行 A 列错误穿越。"""
    if direction == 1:
        return ((bits & ~H_FILE) << 1) & FULL
    if direction == -1:
        return (bits & ~A_FILE) >> 1
    if direction == 8:
        return (bits << 8) & FULL
    if direction == -8:
        return bits >> 8
    if direction == 9:
        return ((bits & ~H_FILE) << 9) & FULL
    if direction == 7:
        return ((bits & ~A_FILE) << 7) & FULL
    if direction == -7:
        return (bits & ~H_FILE) >> 7
    return (bits & ~A_FILE) >> 9


DIRECTIONS = (1, -1, 8, -8, 9, 7, -7, -9)


def legal_bits(own: int, other: int) -> int:
    """从己方棋子向外扩展连续敌方棋链，一次生成所有合法着法。"""
    empty = FULL ^ (own | other)
    moves = 0
    for direction in DIRECTIONS:
        ray = shift(own, direction) & other
        for _ in range(5):
            ray |= shift(ray, direction) & other
        moves |= shift(ray, direction) & empty
    return moves


def flips_for(own: int, other: int, move: int) -> int:
    """收集八个方向全部被夹住的棋子，不能选择性翻子。"""
    result = 0
    for direction in DIRECTIONS:
        current = shift(1 << move, direction)
        captured = 0
        while current & other:
            captured |= current
            current = shift(current, direction)
        if current & own:
            result |= captured
    return result


@dataclass(frozen=True)
class Board:
    black: int = (1 << 28) | (1 << 35)  # E4、D5。
    white: int = (1 << 27) | (1 << 36)  # D4、E5。
    turn: int = 1  # 1 为黑方，-1 为白方。

    @property
    def own_other(self) -> Tuple[int, int]:
        return (self.black, self.white) if self.turn == 1 else (self.white, self.black)

    @property
    def empty_count(self) -> int:
        return 64 - count(self.black | self.white)

    def legal(self) -> List[int]:
        return indices(legal_bits(*self.own_other))

    def terminal(self) -> bool:
        return not legal_bits(self.black, self.white) and not legal_bits(self.white, self.black)

    def play(self, move: Optional[int]) -> "Board":
        own, other = self.own_other
        if move is None:
            if legal_bits(own, other) or self.terminal():
                raise ValueError("仅无合法着法且棋局未结束时允许跳过")
            return Board(self.black, self.white, -self.turn)
        if not 0 <= move < 64 or (own | other) & (1 << move):
            raise ValueError("该位置不能落子")
        captured = flips_for(own, other, move)
        if not captured:
            raise ValueError("落子必须至少翻转一枚对手棋子")
        own |= captured | (1 << move)
        other &= ~captured
        black, white = (own, other) if self.turn == 1 else (other, own)
        return Board(black, white, -self.turn)

    def cells(self) -> List[int]:
        return [
            1 if self.black & (1 << i) else -1 if self.white & (1 << i) else 0 for i in range(64)
        ]

    def winner(self) -> int:
        difference = count(self.black) - count(self.white)
        return (difference > 0) - (difference < 0)


def coordinate(move: Optional[int]) -> str:
    return "PASS" if move is None else chr(65 + move % 8) + str(move // 8 + 1)


def stable_edges(bits: int) -> int:
    """保守稳定子：只计从已占角点连续延伸的同色边线，避免虚报稳定子。"""
    stable = 0
    for corner, steps in ((0, (1, 8)), (7, (-1, 8)), (56, (1, -8)), (63, (-1, -8))):
        if not bits & (1 << corner):
            continue
        stable |= 1 << corner
        for step in steps:
            for distance in range(1, 8):
                index = corner + step * distance
                if not bits & (1 << index):
                    break
                stable |= 1 << index
    return stable


def evaluate(board: Board) -> int:
    """从当前行棋方看局势；两种确定性基线共用同一阶段性评估函数。"""
    own, other = board.own_other
    own_moves, other_moves = legal_bits(own, other), legal_bits(other, own)
    difference = count(own) - count(other)
    if not own_moves and not other_moves:
        return (
            (100000 + difference)
            if difference > 0
            else (-100000 + difference) if difference < 0 else 0
        )
    empty = FULL ^ (own | other)
    frontier = 0
    for direction in DIRECTIONS:
        frontier |= shift(empty, direction)
    phase = (60 - board.empty_count) / 60
    positional = sum(WEIGHTS[i] for i in indices(own)) - sum(WEIGHTS[i] for i in indices(other))
    mobility = count(own_moves) - count(other_moves)
    corners = count(own & CORNERS) - count(other & CORNERS)
    stable = count(stable_edges(own)) - count(stable_edges(other))
    exposed = count(other & frontier) - count(own & frontier)
    # 子数在开局价值低，接近终局才增加权重；行动力和稳定子持续重要。
    return int(
        positional * (1 - phase * 0.5)
        + mobility * (22 - 12 * phase)
        + corners * 160
        + stable * 18
        + exposed * (9 - 6 * phase)
        + difference * (1 + 18 * phase**4)
    )

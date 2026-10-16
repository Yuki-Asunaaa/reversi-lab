"""迭代算法的独立规则对照、固定深度等价与残局检查。"""

import random
import threading
import unittest

from reversi.board import Board, FULL, evaluate
from reversi.experiments import opening
from reversi.advanced import advanced, evaluation, legal, play
from reversi.search import deterministic, search


class AdvancedTests(unittest.TestCase):
    def test_v3_aspiration_matches_independent_full_window(self):
        """用原规则对象独立搜索，验证 V3 窗口失败重搜不会改变根评分。"""

        def reference(board, depth, alpha=-200000, beta=200000):
            moves = board.legal()
            us, them = board.own_other
            if board.terminal():
                return evaluation(us, them, 0, 0, board.empty_count, 3)
            if not moves:
                return -reference(board.play(None), depth, -beta, -alpha)
            if depth == 0:
                return evaluation(us, them, legal(us, them), legal(them, us), board.empty_count, 3)
            best = -200000
            for move in moves:
                value = -reference(board.play(move), depth - 1, -beta, -alpha)
                best = max(best, value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best

        for seed in range(8):
            board = opening(seed + 1550, 10 + seed * 4)
            result = advanced(board, 10, version=3, max_depth=4)
            self.assertEqual(result.depth, 4)
            self.assertEqual(result.score, reference(board, 4))

    def test_v3_evaluation_is_color_symmetric(self):
        """交换己方和对方后估值必须取反，避免先后手视角偏置。"""
        for seed in range(30):
            board = opening(seed, seed + 8)
            us, them = board.own_other
            value = evaluation(us, them, legal(us, them), legal(them, us), board.empty_count, 3)
            inverse = evaluation(them, us, legal(them, us), legal(us, them), board.empty_count, 3)
            self.assertEqual(value, -inverse)

    def test_fast_engine_matches_reference(self):
        """完整随机棋局逐着核对全部合法分支，防止优化引入穿边或漏翻子。"""
        rng = random.Random(7531)
        for _ in range(30):
            board = Board()
            while not board.terminal():
                us, them = board.own_other
                moves = board.legal()
                self.assertEqual(legal(us, them), sum(1 << m for m in moves))
                self.assertEqual(
                    evaluation(us, them, legal(us, them), legal(them, us), board.empty_count, 2),
                    evaluate(board),
                )
                for move in moves:
                    self.assertEqual(play(us, them, move), board.play(move).own_other)
                board = board.play(rng.choice(moves) if moves else None)

    def test_pvs_equals_full_window_at_same_depth(self):
        """V2 共用基线评估：缓存与零窗口不能改变固定深度的最佳分数。"""
        for seed in range(12):
            board = opening(seed + 300, 8 + seed * 3)
            reference = deterministic(board, 10, max_depth=4)
            candidate = advanced(board, 10, version=2, max_depth=4)
            self.assertEqual(candidate.depth, 4)
            self.assertEqual(candidate.score, reference.score)
            # 不强求等分着法相同；验证所选着法确实达到根最佳分数。
            child = deterministic(board.play(candidate.move), 10, max_depth=3)
            self.assertEqual(-child.score, candidate.score)

    def test_exact_endgames_and_pass(self):
        """两版对小残局都应与无剪枝穷举一致，包括强制 PASS。"""
        for seed in range(10):
            board = opening(seed + 800, 54)
            reference = deterministic(board, 10, pruning=False)
            for version in (2, 3):
                candidate = advanced(board, 10, version=version)
                self.assertTrue(candidate.exact)
                self.assertEqual(candidate.score, reference.score)
        board = Board(2, FULL ^ 3, 1)
        self.assertEqual(board.legal(), [])
        self.assertFalse(board.terminal())
        for version in (2, 3):
            result = advanced(board, 1, version=version)
            self.assertIsNone(result.move)
            self.assertTrue(result.exact)
            self.assertLess(result.score, -100000)

    def test_cancel_keeps_legal_fallback(self):
        cancel = threading.Event()
        cancel.set()
        for algorithm in ("hybrid_v2", "hybrid_v3"):
            result = search(Board(), algorithm, 1, cancel=cancel)
            self.assertIn(result.move, Board().legal())
            self.assertEqual(result.depth, 0)


if __name__ == "__main__":
    unittest.main()

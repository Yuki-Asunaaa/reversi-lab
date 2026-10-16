"""独立矩阵规则对照、搜索等价性、时限与校准测试。"""

import json
import random
import threading
import time
import unittest

from reversi.analysis import MODEL_PATH, analyze, predict
from reversi.board import Board, FULL, count, evaluate
from reversi.experiments import opening, play_game, summarize
from reversi.search import ALGORITHMS, deterministic, search
from reversi.session import Session


def reference_moves(cells, color):
    """独立的逐格矩阵实现，验证位棋盘而非复制其移位逻辑。"""
    legal = {}
    for index, value in enumerate(cells):
        if value:
            continue
        row, column = divmod(index, 8)
        flips = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc):
                    continue
                r, c = row + dr, column + dc
                line = []
                while 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == -color:
                    line.append(r * 8 + c)
                    r, c = r + dr, c + dc
                if line and 0 <= r < 8 and 0 <= c < 8 and cells[r * 8 + c] == color:
                    flips += line
        if flips:
            legal[index] = flips
    return legal


class RulesTests(unittest.TestCase):
    def test_initial_and_illegal(self):
        board = Board()
        self.assertEqual(board.legal(), [19, 26, 37, 44])
        for move in (0, 27, -1, 64, None):
            with self.assertRaises(ValueError):
                board.play(move)
        moved = board.play(19)
        self.assertEqual((count(moved.black), count(moved.white)), (4, 1))
        self.assertEqual(moved.turn, -1)

    def test_all_eight_directions(self):
        own = other = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    other |= 1 << ((3 + dr) * 8 + 3 + dc)
                    own |= 1 << ((3 + 2 * dr) * 8 + 3 + 2 * dc)
        board = Board(own, other, 1).play(27)
        self.assertEqual(count(board.black), 17)
        self.assertEqual(board.white, 0)

    def test_forced_pass_and_terminal(self):
        board = Board(FULL ^ 3, 2, -1)
        self.assertEqual(board.legal(), [])
        self.assertFalse(board.terminal())
        board = board.play(None)
        self.assertEqual(board.legal(), [0])
        board = board.play(0)
        self.assertTrue(board.terminal())
        self.assertEqual(board.winner(), 1)
        self.assertTrue(Board(1, 0, -1).terminal())  # 未填满也可结束。

    def test_against_independent_matrix_in_50_games(self):
        rng = random.Random(31)
        passes = 0
        for _ in range(50):
            board = Board()
            while not board.terminal():
                cells = board.cells()
                expected = reference_moves(cells, board.turn)
                self.assertEqual(board.legal(), sorted(expected))
                if not expected:
                    passes += 1
                    board = board.play(None)
                    continue
                move = rng.choice(list(expected))
                for flipped in expected[move] + [move]:
                    cells[flipped] = board.turn
                board = board.play(move)
                self.assertEqual(board.cells(), cells)
                self.assertFalse(board.black & board.white)
            self.assertFalse(reference_moves(board.cells(), 1))
            self.assertFalse(reference_moves(board.cells(), -1))
        self.assertGreater(passes, 0)


class SearchTests(unittest.TestCase):
    def test_alpha_beta_matches_minimax_at_same_depth(self):
        pruned = False
        for seed in range(8):
            board = opening(seed, 14)
            plain = deterministic(board, 10, False, max_depth=3)
            ab = deterministic(board, 10, True, max_depth=3)
            self.assertEqual(plain.depth, 3)
            self.assertEqual(ab.depth, 3)
            self.assertEqual(plain.score, ab.score)
            self.assertLessEqual(ab.nodes, plain.nodes)
            pruned |= ab.nodes < plain.nodes
        self.assertTrue(pruned)

    def test_exact_endgame(self):
        board = opening(55, 55)
        plain = deterministic(board, 10, False)
        ab = deterministic(board, 10, True)
        self.assertTrue(plain.exact)
        self.assertTrue(ab.exact)
        self.assertEqual(ab.score, plain.score)
        hybrid = search(board, "hybrid", 0.2)
        self.assertIn("残局", hybrid.strategy)

    def test_deadlines_and_cancellation(self):
        for algorithm in ALGORITHMS:
            started = time.perf_counter()
            result = search(Board(), algorithm, 0.05)
            self.assertLess(time.perf_counter() - started, 0.25)
            self.assertIn(result.move, Board().legal())
            if algorithm in ("mcts", "hybrid"):
                self.assertGreater(result.iterations, 0)
            cancel = threading.Event()
            cancel.set()
            started = time.perf_counter()
            search(Board(), algorithm, 10, cancel=cancel)
            self.assertLess(time.perf_counter() - started, 0.1)

    def test_mcts_one_move_win(self):
        board = Board(FULL ^ 3, 2, 1)
        result = search(board, "mcts", 0.02)
        self.assertEqual(result.move, 0)
        self.assertEqual(result.score, 1.0)

    def test_experiment_replay_and_summary(self):
        initial = opening(8, 52)
        game = play_game("alphabeta", "mcts", 0.005, initial, 9)
        board = initial
        for move in game["history"]:
            self.assertEqual(move["color"], board.turn)
            board = board.play(move["move"])
        self.assertTrue(board.terminal())
        self.assertEqual(game["winner"], board.winner())
        rows = summarize([game])
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["wins"] for row in rows), int(bool(game["winner"])))


class AnalysisTests(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.exists(), "尚未生成校准模型")
    def test_calibrated_probabilities(self):
        model = json.loads(MODEL_PATH.read_text())
        for score in (-1000, -200, 0, 300, 2000):
            p = predict(score, 20, model)
            mirror = predict(-score, 20, model)
            self.assertAlmostEqual(sum(p.values()), 1)
            self.assertAlmostEqual(p["black"], mirror["white"], places=8)
            self.assertTrue(all(0 <= value <= 1 for value in p.values()))
        result = analyze(Board())
        self.assertEqual(result["status"], "provisional")

    def test_exact_analysis_is_not_probability(self):
        result = analyze(Board(FULL ^ 3, 2, 1))
        self.assertEqual(result["status"], "solved")
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["outcome"], 1)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()

    def tearDown(self):
        self.session.close()

    def test_timeout_and_pause_clock(self):
        s = self.session
        with s.lock:
            s.status, s.phase = "running", "thinking"
            s.active_config = {"human": True}
            s.turn_started = time.monotonic() - 61
            s.tick()
            self.assertEqual(s.status, "finished")
            self.assertEqual(s.outcome, -1)
            self.assertIn("一分钟", s.reason)
            s.reset()
            s.status, s.phase = "running", "thinking"
            s.active_config = {"human": True}
            s.turn_started = time.monotonic() - 5
            s.command("pause", {})
            elapsed = s.turn_elapsed()
            self.assertAlmostEqual(s.turn_elapsed(), elapsed, places=5)
            self.assertAlmostEqual(s.state()["totals"]["black"], elapsed, places=5)
            s.command("start", {})
            self.assertGreaterEqual(s.turn_elapsed(), elapsed)

    def test_config_changes_do_not_override_current_player(self):
        s = self.session
        with s.lock:
            s.active_config = {"algorithm": "alphabeta", "seconds": 3, "human": False}
            s.command("config", {"black": {"algorithm": "mcts", "seconds": 0.2}})
            self.assertEqual(s.active_config["algorithm"], "alphabeta")
            self.assertEqual(s.config["black"]["algorithm"], "mcts")


if __name__ == "__main__":
    unittest.main()

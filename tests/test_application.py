"""真实 HTTP 与独立搜索进程的端到端检查，使用临时实验目录。"""

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from main import make_handler
from reversi.board import Board
from reversi.session import Session


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = patch("reversi.session.ROOT", Path(cls.temporary.name))
        cls.paths.start()
        cls.session = Session()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.session))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.session.close()
        cls.paths.stop()
        cls.temporary.cleanup()

    def post(self, action, payload=None):
        request = urllib.request.Request(
            self.url + "/api/" + action,
            json.dumps(payload or {}).encode(),
            {"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(request, timeout=3))

    def get(self, path="/api/state"):
        return json.load(urllib.request.urlopen(self.url + path, timeout=3))

    def wait(self, predicate, timeout=40):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get()
            self.assertFalse(state["error"], state["error"])
            if predicate(state):
                return state
            time.sleep(0.04)
        self.fail("应用状态未在时限内满足条件")

    def test_pve_new_game_cancels_previous_worker(self):
        self.post("new")
        self.post(
            "config",
            {"mode": "pve", "human_color": 1, "white": {"algorithm": "alphabeta", "seconds": 0.05}},
        )
        self.post("start")
        self.wait(lambda state: state["phase"] == "thinking")
        with self.assertRaises(urllib.error.HTTPError):
            self.post("move", {"move": 0})
        self.post("move", {"move": 19})
        state = self.wait(lambda state: len(state["history"]) >= 2)
        self.assertEqual(state["history"][0]["coordinate"], "D3")
        self.assertEqual(state["history"][1]["algorithm"], "alphabeta")
        self.post("new")
        self.post("config", {"mode": "eve", "black": {"algorithm": "mcts", "seconds": 3}})
        self.post("start")
        self.wait(lambda state: state["phase"] == "thinking")
        self.post("new")
        time.sleep(0.15)
        state = self.get()
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["history"], [])
        self.assertEqual(state["counts"], {"black": 2, "white": 2})

    def test_eve_complete_and_export_replay(self):
        self.post("new")
        self.post(
            "config",
            {
                "mode": "eve",
                "black": {"algorithm": "hybrid", "seconds": 0.05},
                "white": {"algorithm": "minimax", "seconds": 0.05},
            },
        )
        self.post("start")
        state = self.wait(lambda state: state["status"] == "finished")
        board = Board()
        data = self.get("/api/export/game")
        for move in data["history"]:
            self.assertEqual(board.turn, move["color"])
            board = board.play(move["move"])
        self.assertTrue(board.terminal())
        self.assertEqual(board.cells(), data["final_cells"])
        self.assertEqual(board.winner(), state["outcome"])
        self.assertTrue(any("残局" in move["strategy"] for move in data["history"]))

    def test_experiment_cancel_complete_and_exports(self):
        self.post("new")
        config = {"algorithms": ["minimax", "alphabeta"], "budgets": [0.05], "pairs": 1}
        self.post("experiment", config)
        self.post("stop_experiment")
        self.wait(lambda state: state["experiment"]["status"] == "cancelled")
        self.post("experiment", config)
        state = self.wait(lambda state: state["experiment"]["status"] == "completed")
        self.assertEqual(state["experiment"]["completed"], 2)
        data = self.get("/api/export/experiment")
        self.assertEqual(len(data["games"]), 2)
        self.assertEqual(data["games"][0]["initial"], data["games"][1]["initial"])
        self.assertEqual(data["games"][0]["black"], data["games"][1]["white"])
        csv = urllib.request.urlopen(self.url + "/api/export/csv").read().decode("utf-8-sig")
        self.assertIn("win_rate", csv)
        self.assertIn("alphabeta", csv)

    def test_paused_ai_updates_probability_for_the_new_board(self):
        self.post("new")
        self.post("config", {"mode": "eve", "black": {"algorithm": "alphabeta", "seconds": 0.2}})
        self.post("start")
        self.wait(lambda state: state["phase"] == "thinking")
        self.post("pause")
        state = self.wait(
            lambda state: state["status"] == "paused"
            and state["analysis"]["status"] == "calibrated"
        )
        self.assertGreaterEqual(len(state["history"]), 1)
        exported = self.get("/api/export/game")
        self.assertEqual(exported["analysis_history"][-1]["ply"], len(state["history"]))
        self.assertEqual(state["turn_elapsed"], 0)


if __name__ == "__main__":
    unittest.main()

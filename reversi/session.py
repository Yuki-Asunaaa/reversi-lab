"""对局协调：主线程维护状态，独立进程搜索，使用单调时钟执行超时规则。"""

import copy
import json
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from .analysis import analysis_worker
from .board import Board, coordinate, count
from .experiments import experiment_worker
from .search import ALGORITHMS, worker

ROOT = Path(__file__).resolve().parent.parent


class Task:
    """统一管理可取消的子进程，避免新开一局后旧 AI 结果污染棋盘。"""

    def __init__(self, target, *args):
        context = mp.get_context("spawn")
        self.queue, self.cancel = context.Queue(), context.Event()
        self.process = context.Process(
            target=target, args=(*args, self.queue, self.cancel), daemon=True
        )
        self.started = time.monotonic()
        self.process.start()

    def messages(self):
        messages = []
        while True:
            try:
                messages.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def stop(self):
        self.cancel.set()
        self.process.join(timeout=0.05)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.1)
        self.queue.close()


class Session:
    def __init__(self, persist_experiment=True):
        self.lock = threading.RLock()
        self.persist_experiment = persist_experiment
        self.ai_task = self.analysis_task = self.experiment_task = None
        self.experiment = {"status": "idle", "summary": [], "completed": 0, "total": 0}
        saved = ROOT / "results" / "latest_experiment.json"
        if self.persist_experiment and saved.exists():
            try:
                self.experiment = json.loads(saved.read_text(encoding="utf-8"))
                if self.experiment.get("status") == "running":
                    self.experiment["status"] = "cancelled"
            except (ValueError, OSError):
                pass
        self.config = {
            "mode": "pve",
            "human_color": 1,
            "black": {"algorithm": "hybrid_v3", "seconds": 3.0},
            "white": {"algorithm": "hybrid_v3", "seconds": 3.0},
        }
        self.closed = False
        self.reset()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop_task(self, name):
        task = getattr(self, name, None)
        if task:
            task.stop()
            setattr(self, name, None)

    def reset(self):
        """回到标准初始棋局，同时终止旧的搜索和分析任务。"""
        self.stop_task("ai_task")
        self.stop_task("analysis_task")
        self.board = Board()
        self.status, self.phase = "idle", "ready"
        self.history, self.analysis_history = [], []
        self.analysis = {"status": "idle", "probabilities": None, "label": "开始对局后自动分析"}
        self.progress = {}
        self.last_move = None
        self.totals = {1: 0.0, -1: 0.0}
        self.turn_started = None
        self.turn_accumulated = 0.0
        self.active_config = None
        self.pause_requested = False
        self.outcome, self.reason, self.error = None, "", ""
        self.revision = getattr(self, "revision", 0) + 1

    def turn_elapsed(self):
        """暂停前累计值加上本次运行时间；只使用不受系统校时影响的单调时钟。"""
        return self.turn_accumulated + (
            time.monotonic() - self.turn_started if self.turn_started is not None else 0.0
        )

    def begin_turn(self):
        """先判终局和强制 PASS，再分析新局面，最后才启动落子计时。"""
        self.stop_task("analysis_task")
        if self.board.terminal():
            self.finish(
                self.board.winner(),
                "棋盘已满" if self.board.empty_count == 0 else "双方均无合法落子",
            )
            return
        if not self.board.legal():
            self.history.append(
                {
                    "ply": len(self.history) + 1,
                    "color": self.board.turn,
                    "move": None,
                    "coordinate": "PASS",
                    "elapsed": 0,
                    "algorithm": "pass",
                    "seconds": 0,
                    "nodes": 0,
                    "depth": 0,
                    "iterations": 0,
                    "strategy": "无合法着法，自动跳过",
                }
            )
            self.board = self.board.play(None)
            self.revision += 1
        self.turn_started = None
        self.turn_accumulated = 0
        self.active_config = None
        self.progress = {}
        self.prepare_analysis()

    def prepare_analysis(self):
        """局面变化后立即废弃旧胜率；暂停的对局也可以完成新局面分析。"""
        self.stop_task("analysis_task")
        self.phase = "analysis"
        self.analysis = {"status": "pending", "probabilities": None, "label": "统一分析当前局面…"}
        self.analysis_task = Task(analysis_worker, self.board, copy.deepcopy(self.config))

    def activate(self):
        """锁定本回合的控制者、算法和时限，后续设置变化留到下一回合。"""
        self.phase = "thinking"
        self.active_config = copy.deepcopy(
            self.config["black" if self.board.turn == 1 else "white"]
        )
        human = self.config["mode"] == "pve" and self.config["human_color"] == self.board.turn
        self.active_config["human"] = human
        self.turn_started = time.monotonic()
        if not human:
            # 分析与搜索串行：统一分析的额外计算不侵占任一 AI 的搜索时间。
            budget = max(0.001, self.active_config["seconds"] - 0.01)
            self.ai_task = Task(worker, self.board, self.active_config["algorithm"], budget)

    def commit(self, move, result=None):
        """再次校验落子与超时，写入棋谱，并推进到下一个合法回合。"""
        elapsed = self.turn_elapsed()
        if elapsed > 60:
            self.finish(-self.board.turn, "单步超过一分钟，判超时负")
            return
        color = self.board.turn
        updated = self.board.play(move)
        config = self.active_config or {}
        result = result or {}
        self.history.append(
            {
                "ply": len(self.history) + 1,
                "color": color,
                "move": move,
                "coordinate": coordinate(move),
                "elapsed": elapsed,
                "search_elapsed": result.get("elapsed", 0),
                "seconds": config.get("seconds", 0),
                "algorithm": "human" if config.get("human") else config.get("algorithm", "human"),
                "nodes": result.get("nodes", 0),
                "depth": result.get("depth", 0),
                "iterations": result.get("iterations", 0),
                "strategy": result.get("strategy", "人类落子"),
                "exact": result.get("exact", False),
            }
        )
        self.totals[color] += elapsed
        self.board, self.last_move = updated, move
        self.turn_started = None
        self.turn_accumulated = 0.0
        self.revision += 1
        self.active_config = None
        if self.board.terminal():
            self.finish(
                self.board.winner(),
                "棋盘已满" if self.board.empty_count == 0 else "双方均无合法落子",
            )
        elif self.pause_requested:
            self.status, self.phase = "paused", "ready"
            self.pause_requested = False
            self.prepare_analysis()
        else:
            self.begin_turn()

    def finish(self, outcome, reason):
        """统一处理自然终局和超时失败，停止任务并冻结累计计时。"""
        self.stop_task("ai_task")
        self.stop_task("analysis_task")
        if self.turn_started is not None:
            self.totals[self.board.turn] += self.turn_elapsed()
        self.status, self.phase = "finished", "finished"
        self.turn_started = None
        self.turn_accumulated = 0
        self.outcome, self.reason = outcome, reason
        self.analysis = {
            "status": "terminal",
            "probabilities": {
                "black": float(outcome == 1),
                "draw": float(outcome == 0),
                "white": float(outcome == -1),
            },
            "label": "棋局实际结果",
            "exact": True,
        }
        self.revision += 1

    def tick(self):
        """非阻塞读取进程消息，保证长时间 AI 搜索期间接口仍能响应。"""
        if self.experiment_task:
            for message in self.experiment_task.messages():
                kind = message["type"]
                if kind == "move":
                    self.experiment["current"] = message["data"]
                elif kind in ("snapshot", "done"):
                    self.experiment = message["data"]
                    # 本地模式按局原子落盘；公开服务按访客隔离，避免写入共享实例目录。
                    if self.persist_experiment:
                        (ROOT / "results").mkdir(exist_ok=True)
                        temporary = ROOT / "results" / "latest_experiment.tmp"
                        temporary.write_text(
                            json.dumps(self.experiment, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        temporary.replace(ROOT / "results" / "latest_experiment.json")
                    if kind == "done":
                        self.stop_task("experiment_task")
                        break
                elif kind == "error":
                    self.experiment["status"] = "error"
                    self.experiment["error"] = message["message"]
                    self.stop_task("experiment_task")
                    break
            return
        if self.status not in ("running", "paused"):
            return
        if self.analysis_task:
            messages = self.analysis_task.messages()
            if messages:
                self.analysis = messages[-1]
                self.analysis_history.append({"ply": len(self.history), "analysis": self.analysis})
                self.stop_task("analysis_task")
                if self.status == "running":
                    self.activate()
                else:
                    self.phase = "ready"
            elif time.monotonic() - self.analysis_task.started > 3:
                self.stop_task("analysis_task")
                self.analysis = {
                    "status": "provisional",
                    "probabilities": None,
                    "label": "分析超时，继续对局",
                }
                if self.status == "running":
                    self.activate()
                else:
                    self.phase = "ready"
        if self.status == "paused":
            return
        if self.turn_started is not None and self.turn_elapsed() > 60:
            self.finish(-self.board.turn, "单步超过一分钟，判超时负")
            return
        if self.ai_task:
            for message in self.ai_task.messages():
                if message["type"] == "progress":
                    self.progress = message["data"]
                elif message["type"] == "done":
                    self.stop_task("ai_task")
                    self.progress = message["data"]
                    self.commit(message["data"]["move"], message["data"])
                    break
                elif message["type"] == "error":
                    self.stop_task("ai_task")
                    self.error = "AI 搜索失败：" + message["message"]
                    self.turn_accumulated = self.turn_elapsed()
                    self.turn_started = None
                    self.status = "paused"
                    break

    def loop(self):
        """协调循环独立于 HTTP 请求；异常时暂停并向界面给出原因。"""
        while not self.closed:
            with self.lock:
                try:
                    self.tick()
                except Exception as error:
                    self.error = str(error)
                    self.status = "paused"
                    self.stop_task("ai_task")
                    self.stop_task("analysis_task")
                    self.turn_accumulated = self.turn_elapsed()
                    self.turn_started = None
            time.sleep(0.025)

    def state(self):
        """生成供界面展示的快照，不让界面直接接触内部棋盘对象。"""
        with self.lock:
            elapsed = self.turn_elapsed()
            totals = dict(self.totals)
            if self.status in ("running", "paused"):
                totals[self.board.turn] += elapsed
            experiment = {key: value for key, value in self.experiment.items() if key != "games"}
            return {
                "cells": self.board.cells(),
                "turn": self.board.turn,
                "legal": self.board.legal(),
                "counts": {"black": count(self.board.black), "white": count(self.board.white)},
                "status": self.status,
                "phase": self.phase,
                "config": self.config,
                "active_config": self.active_config,
                "history": self.history,
                "analysis": self.analysis,
                "progress": self.progress,
                "revision": self.revision,
                "last_move": self.last_move,
                "turn_elapsed": elapsed,
                "totals": {"black": totals[1], "white": totals[-1]},
                "outcome": self.outcome,
                "reason": self.reason,
                "error": self.error,
                "pause_requested": self.pause_requested,
                "experiment": experiment,
            }

    def command(self, action, payload):
        """集中验证所有操作的参数和当前状态，避免非法控制流。"""
        with self.lock:
            if action == "config":
                mode = payload.get("mode", self.config["mode"])
                human = payload.get("human_color", self.config["human_color"])
                if mode not in ("pve", "eve") or human not in (1, -1):
                    raise ValueError("对弈模式或执棋颜色不正确")
                updated = copy.deepcopy(self.config)
                updated.update(mode=mode, human_color=human)
                for color in ("black", "white"):
                    side = payload.get(color, updated[color])
                    algorithm, seconds = side["algorithm"], float(side["seconds"])
                    if algorithm not in ALGORITHMS or not 0.05 <= seconds <= 59.5:
                        raise ValueError("算法无效或时间不在 0.05～59.5 秒之间")
                    updated[color] = {"algorithm": algorithm, "seconds": seconds}
                self.config = updated
                # 配置在分析阶段改变时重新取快照；已经开始思考的回合仍遵守原设置。
                if self.analysis_task and self.phase == "analysis":
                    self.prepare_analysis()
                elif self.status == "paused" and self.phase == "ready":
                    # 暂停在回合边界时同步更新对战预测，不改已开始的人类计时。
                    self.prepare_analysis()
            elif action == "new":
                if self.experiment_task:
                    raise ValueError("请先停止批量实验")
                self.reset()
            elif action == "start":
                if self.experiment_task:
                    raise ValueError("批量实验进行中，请先停止实验")
                if self.status not in ("idle", "paused"):
                    raise ValueError("当前无法开始，请新建一局")
                self.status = "running"
                if (
                    self.active_config
                    and self.active_config.get("human")
                    and self.phase == "thinking"
                ):
                    self.turn_started = time.monotonic()
                else:
                    self.begin_turn()
            elif action == "pause":
                if self.status != "running":
                    return self.state()
                if self.ai_task:
                    self.pause_requested = True
                else:
                    self.turn_accumulated = self.turn_elapsed()
                    self.turn_started = None
                    self.status = "paused"
            elif action == "move":
                if (
                    self.status != "running"
                    or self.phase != "thinking"
                    or not self.active_config
                    or not self.active_config.get("human")
                ):
                    raise ValueError("当前不是人类落子回合")
                move = payload.get("move")
                if not isinstance(move, int) or isinstance(move, bool):
                    raise ValueError("落点格式不正确")
                self.commit(move)
            elif action == "experiment":
                if self.status == "running" or self.experiment_task:
                    raise ValueError("请先暂停当前对局或停止已有实验")
                algorithms = list(dict.fromkeys(payload.get("algorithms", list(ALGORITHMS))))
                budgets = [float(value) for value in payload.get("budgets", [0.2])]
                pairs = int(payload.get("pairs", 1))
                if len(algorithms) < 2 or any(a not in ALGORITHMS for a in algorithms):
                    raise ValueError("至少选择两种算法")
                if (
                    not budgets
                    or len(budgets) > 6
                    or any(not 0.05 <= n <= 30 for n in budgets)
                    or not 1 <= pairs <= 20
                ):
                    raise ValueError("实验档位或对局组数不合法")
                config = {
                    "algorithms": algorithms,
                    "budgets": budgets,
                    "pairs": pairs,
                    "seed": int(payload.get("seed", 20261014)),
                }
                # 暂停后的分析也必须结束，保证批量实验只有一个搜索任务。
                self.stop_task("analysis_task")
                total = len(algorithms) * (len(algorithms) - 1) * len(budgets) * pairs
                self.experiment = {
                    "status": "running",
                    "summary": [],
                    "completed": 0,
                    "total": total,
                    "config": config,
                }
                self.experiment_task = Task(experiment_worker, config)
            elif action == "stop_experiment":
                if self.experiment_task:
                    self.experiment_task.cancel.set()
            else:
                raise ValueError("未知操作")
            return self.state()

    def export_game(self):
        """逐步配置与分析轨迹随棋谱导出，方便核查中途切换过的算法。"""
        with self.lock:
            return {
                "version": 1,
                "config": self.config,
                "history": self.history,
                "analysis_history": self.analysis_history,
                "result": self.outcome,
                "reason": self.reason,
                "final_cells": self.board.cells(),
                "time_note": "elapsed 含搜索进程启动及协调开销；search_elapsed 仅搜索；回合前统一分析不计入落子时间",
            }

    def close(self):
        """释放所有子进程；程序退出后不留下后台 AI 任务。"""
        with self.lock:
            self.closed = True
            for name in ("ai_task", "analysis_task", "experiment_task"):
                self.stop_task(name)

# 算法版本快照

所有尝试均保留，可以在界面选择，或调用 `search(board, algorithm, seconds)`。

- `v1/`：`hybrid`，原 MCTS/Alpha-Beta 阶段切换逻辑；注册表中其他版本的名称仅为统一入口元数据。
- `v2/`：`hybrid_v2`，`advanced(..., version=2)` 分支。
- `v3/`：`hybrid_v3`，`advanced(..., version=3)` 分支。

V2、V3 共用搜索框架，快照保留整个文件及版本参数，便于比较。快照用于审计；实际运行入口是主项目 `reversi/search.py`，不要单独运行快照文件。

每版 `manifest.json` 记录代码 SHA-256。独立验证文件另存执行时源文件哈希。原版既有实验中的 `hybrid` 始终指 V1，不会被新算法覆盖或重新解释。

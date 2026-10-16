/* 全部交互在本机进行；服务端负责规则和计时，前端只负责呈现与发送操作。 */
"use strict";
const $ = (id) => document.getElementById(id);
const ALGORITHMS = {
  minimax: "Minimax",
  alphabeta: "Alpha-Beta",
  mcts: "MCTS",
  hybrid: "V1 · 阶段混合（原版）",
  hybrid_v2: "V2 · 缓存主变化搜索",
  hybrid_v3: "V3 · 动态评估混合",
};
const BUDGETS = [0.05, 0.2, 0.5, 1, 3, 10, 30];
let state = null;
let stateReceivedAt = performance.now();
let settingsInitialized = false;
let selectedMode = "pve";
let requestBusy = false;
let toastTimer;
let lastHistoryKey = "";
let lastExperimentKey = "";
let experimentConfigInitialized = false;
let lastRevision = -1;
let trends = [];
let seenAnalyses = new Set();

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 3400);
}

async function command(action, payload = {}) {
  const response = await fetch(`/api/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作未完成");
  receive(data);
  return data;
}

async function perform(action, payload = {}) {
  try {
    return await command(action, payload);
  } catch (error) {
    toast(error.message);
    return null;
  }
}

function switchView(view) {
  document
    .querySelectorAll(".view")
    .forEach((item) =>
      item.classList.toggle("active", item.id === `view-${view}`),
    );
  document
    .querySelectorAll(".tab")
    .forEach((item) =>
      item.classList.toggle("active", item.dataset.view === view),
    );
  if (view === "about") loadCalibration();
  if (view === "evolution") loadEvolution();
  if (view === "play") resizeBoard();
}

document
  .querySelectorAll(".tab")
  .forEach((button) =>
    button.addEventListener("click", () => switchView(button.dataset.view)),
  );
$("toLab").addEventListener("click", (event) => {
  event.preventDefault();
  switchView("lab");
});
$("calibrationBtn").addEventListener("click", () => {
  switchView("about");
  $("calibrationDetails").scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
});

/* 设置与回合采用明确边界：正在搜索的回合不会因下拉框变化而被取消。 */
for (const color of ["black", "white"]) {
  for (const [key, name] of Object.entries(ALGORITHMS))
    $("" + color + "Algorithm").add(new Option(name, key));
  for (const seconds of BUDGETS)
    $(color + "Time").add(new Option(`${seconds} 秒`, seconds));
  $(color + "Time").add(new Option("自定义…", "custom"));
  $(color + "Algorithm").addEventListener("change", saveSettings);
  $(color + "Time").addEventListener("change", async () => {
    const select = $(color + "Time");
    if (select.value === "custom") {
      const value = window.prompt("输入每步搜索上限（0.05～59.5 秒）", "5");
      const seconds = Number(value);
      if (
        value === null ||
        !Number.isFinite(seconds) ||
        seconds < 0.05 ||
        seconds > 59.5
      ) {
        select.value = String(state.config[color].seconds);
        if (value !== null) toast("请输入 0.05～59.5 之间的秒数");
        return;
      }
      ensureTimeOption(color, seconds);
      select.value = String(seconds);
    }
    await saveSettings();
  });
}

function ensureTimeOption(color, seconds) {
  const select = $(color + "Time");
  if (![...select.options].some((item) => item.value === String(seconds)))
    select.add(new Option(`${seconds} 秒`, seconds), select.options.length - 1);
}

function updateRoleControls() {
  const human = Number($("humanColor").value);
  $("pveBtn").classList.toggle("selected", selectedMode === "pve");
  $("eveBtn").classList.toggle("selected", selectedMode === "eve");
  $("humanColorRow").hidden = selectedMode === "eve";
  for (const [color, value] of [
    ["black", 1],
    ["white", -1],
  ]) {
    const isHuman = selectedMode === "pve" && human === value;
    $(color + "Role").textContent = isHuman ? "玩家" : "AI";
    $(color + "Algorithm")
      .closest(".side-config")
      .classList.toggle("is-human", isHuman);
    $(color + "Algorithm").disabled = isHuman;
    $(color + "Time").disabled = isHuman;
  }
}

async function saveSettings() {
  updateRoleControls();
  const payload = {
    mode: selectedMode,
    human_color: Number($("humanColor").value),
  };
  for (const color of ["black", "white"])
    payload[color] = {
      algorithm: $(color + "Algorithm").value,
      seconds: Number($(color + "Time").value),
    };
  await perform("config", payload);
}
$("pveBtn").addEventListener("click", () => {
  selectedMode = "pve";
  saveSettings();
});
$("eveBtn").addEventListener("click", () => {
  selectedMode = "eve";
  saveSettings();
});
$("humanColor").addEventListener("change", saveSettings);
$("startBtn").addEventListener("click", () => perform("start"));
$("pauseBtn").addEventListener("click", () => perform("pause"));

async function newGame() {
  if (state && (state.history.length || state.status === "running")) {
    const dialog = $("confirmDialog");
    // 原生 dialog 会保留上次返回值；每次重置，确保 Esc 始终表示取消。
    dialog.returnValue = "cancel";
    dialog.showModal();
    const answer = await new Promise((resolve) =>
      dialog.addEventListener("close", () => resolve(dialog.returnValue), {
        once: true,
      }),
    );
    if (answer !== "confirm") return;
  }
  const next = await perform("new");
  if (next) {
    trends = [];
    seenAnalyses.clear();
  }
}
$("newBtn").addEventListener("click", newGame);
$("playAgainBtn").addEventListener("click", async () => {
  await perform("new");
  trends = [];
  seenAnalyses.clear();
  await perform("start");
});

/* 棋盘绘制：支持高 DPI；每个翻转独立记录动画，轮询更新不会重启动画。 */
const canvas = $("board");
const ctx = canvas.getContext("2d");
let boardSize = 560;
let cells = Array(64).fill(0);
let animations = new Map();
let hoverCell = -1;
let boardDirty = true;
let reducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;
if (reducedMotion) $("animate").checked = false;
const cellButtons = [];
for (let index = 0; index < 64; index++) {
  const button = document.createElement("button");
  button.className = "board-cell";
  button.setAttribute("role", "gridcell");
  button.setAttribute("aria-rowindex", Math.floor(index / 8) + 1);
  button.setAttribute("aria-colindex", (index % 8) + 1);
  button.addEventListener("pointerenter", () => {
    hoverCell = index;
    boardDirty = true;
  });
  button.addEventListener("pointerleave", () => {
    hoverCell = -1;
    boardDirty = true;
  });
  button.addEventListener("focus", () => {
    hoverCell = index;
    boardDirty = true;
  });
  button.addEventListener("blur", () => {
    hoverCell = -1;
    boardDirty = true;
  });
  button.addEventListener("click", async () => {
    if (requestBusy) return;
    requestBusy = true;
    await perform("move", { move: index });
    requestBusy = false;
  });
  $("boardGrid").appendChild(button);
  cellButtons.push(button);
}
function coordinate(index) {
  return String.fromCharCode(65 + (index % 8)) + (Math.floor(index / 8) + 1);
}
function resizeBoard() {
  const width = canvas.getBoundingClientRect().width;
  if (!width) return;
  boardSize = width;
  const ratio = Math.min(window.devicePixelRatio || 1, 3);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(width * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  boardDirty = true;
}
new ResizeObserver(resizeBoard).observe(canvas);
$("showHints").addEventListener("change", () => {
  boardDirty = true;
});
$("animate").addEventListener("change", () => {
  animations.clear();
  boardDirty = true;
});

function updateBoard(nextCells) {
  const now = performance.now();
  if (state.status === "idle") animations.clear();
  for (let index = 0; index < 64; index++) {
    if (
      cells[index] !== nextCells[index] &&
      nextCells[index] &&
      $("animate").checked &&
      lastRevision >= 0 &&
      state.status !== "idle"
    ) {
      animations.set(index, {
        from: cells[index],
        to: nextCells[index],
        start: now,
        duration: cells[index] ? 430 : 330,
      });
    }
  }
  cells = nextCells.slice();
  boardDirty = true;
}
function drawDisc(x, y, radius, color, scaleX = 1, opacity = 1, lift = 0) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.translate(x, y - lift);
  ctx.scale(Math.max(0.045, scaleX), 1);
  ctx.shadowColor = "rgba(0, 12, 4, 0.42)";
  ctx.shadowBlur = radius * 0.31;
  ctx.shadowOffsetY = radius * 0.14 + lift * 0.3;
  const gradient = ctx.createRadialGradient(
    -radius * 0.33,
    -radius * 0.4,
    radius * 0.05,
    0,
    0,
    radius * 1.04,
  );
  if (color === 1) {
    gradient.addColorStop(0, "#68716a");
    gradient.addColorStop(0.32, "#353e38");
    gradient.addColorStop(0.72, "#202a24");
    gradient.addColorStop(0.95, "#111c15");
    gradient.addColorStop(1, "#415047");
  } else {
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(0.43, "#faf9ef");
    gradient.addColorStop(0.8, "#e0e2d5");
    gradient.addColorStop(0.97, "#b8c0ae");
    gradient.addColorStop(1, "#f5f6e9");
  }
  ctx.beginPath();
  ctx.arc(0, 0, radius, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.beginPath();
  ctx.arc(0, 0, radius * 0.89, -Math.PI * 0.92, -Math.PI * 0.17);
  ctx.strokeStyle = color === 1 ? "#a5b2a62d" : "#ffffffba";
  ctx.lineWidth = radius * 0.035;
  ctx.stroke();
  ctx.restore();
}

function drawBoard(now) {
  const size = boardSize,
    margin = size * 0.065,
    step = (size * 0.87) / 8;
  ctx.clearRect(0, 0, size, size);
  const background = ctx.createLinearGradient(0, 0, size, size);
  background.addColorStop(0, "#345641");
  background.addColorStop(0.6, "#294834");
  background.addColorStop(1, "#233e2e");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "#91a57b25";
  ctx.lineWidth = 1;
  ctx.strokeRect(7, 7, size - 14, size - 14);
  ctx.strokeStyle = "#081f1535";
  ctx.strokeRect(margin - 3, margin - 3, step * 8 + 6, step * 8 + 6);
  for (let row = 0; row < 8; row++)
    for (let col = 0; col < 8; col++) {
      ctx.fillStyle = (row + col) % 2 ? "#89a37906" : "#0c2b1809";
      ctx.fillRect(margin + col * step, margin + row * step, step, step);
    }
  ctx.strokeStyle = "#0d291c69";
  ctx.lineWidth = 0.9;
  for (let n = 0; n <= 8; n++) {
    ctx.beginPath();
    ctx.moveTo(margin + n * step, margin);
    ctx.lineTo(margin + n * step, size - margin);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin, margin + n * step);
    ctx.lineTo(size - margin, margin + n * step);
    ctx.stroke();
  }
  ctx.font = `${size * 0.019}px "Avenir Next", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#b4c1a480";
  for (let n = 0; n < 8; n++) {
    const center = margin + (n + 0.5) * step;
    ctx.fillText(String.fromCharCode(65 + n), center, margin * 0.48);
    ctx.fillText(String.fromCharCode(65 + n), center, size - margin * 0.46);
    ctx.fillText(n + 1, margin * 0.48, center);
    ctx.fillText(n + 1, size - margin * 0.48, center);
  }
  for (const [r, c] of [
    [2, 2],
    [2, 6],
    [6, 2],
    [6, 6],
  ]) {
    ctx.beginPath();
    ctx.arc(margin + c * step, margin + r * step, 2.1, 0, Math.PI * 2);
    ctx.fillStyle = "#152f20";
    ctx.fill();
  }
  const legal = state ? state.legal : [];
  const canPlay =
    state &&
    state.status === "running" &&
    state.phase === "thinking" &&
    state.active_config?.human;
  if ($("showHints").checked && state && state.status !== "finished")
    for (const index of legal) {
      const x = margin + ((index % 8) + 0.5) * step,
        y = margin + (Math.floor(index / 8) + 0.5) * step;
      ctx.beginPath();
      ctx.arc(x, y, step * 0.11, 0, Math.PI * 2);
      ctx.strokeStyle = canPlay ? "#c5d4a77d" : "#c5d4a73b";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = "#b8c89417";
      ctx.fill();
      if (index === hoverCell && canPlay)
        drawDisc(x, y, step * 0.405, state.turn, 1, 0.42);
    }
  for (let index = 0; index < 64; index++) {
    if (!cells[index]) continue;
    const x = margin + ((index % 8) + 0.5) * step,
      y = margin + (Math.floor(index / 8) + 0.5) * step;
    let color = cells[index],
      scale = 1,
      radius = step * 0.405,
      lift = 0;
    const animation = animations.get(index);
    if (animation) {
      const t = Math.min(1, (now - animation.start) / animation.duration);
      if (animation.from) {
        scale = Math.abs(Math.cos(Math.PI * t));
        color = t < 0.5 ? animation.from : animation.to;
        lift = Math.sin(Math.PI * t) * step * 0.08;
      } else {
        radius *= 0.7 + 0.3 * (1 - Math.pow(1 - t, 3));
        lift = (1 - t) ** 2 * step * 0.25;
      }
      if (t === 1) animations.delete(index);
    }
    drawDisc(x, y, radius, color, scale, 1, lift);
    if (index === state?.last_move && !animation) {
      ctx.beginPath();
      ctx.arc(x, y, step * 0.055, 0, Math.PI * 2);
      ctx.strokeStyle = color === 1 ? "#b9c49e" : "#6f8762";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
  }
  boardDirty = false;
}

/* 状态展示：服务器是唯一真值，客户端插值仅用于流畅显示计时。 */
function receive(next) {
  const was = state;
  state = next;
  stateReceivedAt = performance.now();
  if (!settingsInitialized) {
    selectedMode = next.config.mode;
    $("humanColor").value = next.config.human_color;
    for (const color of ["black", "white"]) {
      ensureTimeOption(color, next.config[color].seconds);
      $(color + "Algorithm").value = next.config[color].algorithm;
      $(color + "Time").value = next.config[color].seconds;
    }
    updateRoleControls();
    settingsInitialized = true;
  }
  if (next.revision !== lastRevision) {
    if (next.history.length < (was?.history.length || 0)) {
      trends = [];
      seenAnalyses.clear();
    }
    updateBoard(next.cells);
    lastRevision = next.revision;
  }
  $("connectionDot").style.background = "#668367";
  $("connectionText").textContent = "本地引擎已连接";
  for (const [color, value] of [
    ["black", 1],
    ["white", -1],
  ]) {
    $(color + "Count").textContent = next.counts[color];
    const configuredHuman =
      next.config.mode === "pve" && next.config.human_color === value;
    const active = next.turn === value && next.active_config;
    $(color + "ModelName").textContent = active
      ? next.active_config.human
        ? "玩家"
        : ALGORITHMS[next.active_config.algorithm]
      : configuredHuman
        ? "玩家"
        : ALGORITHMS[next.config[color].algorithm];
    $(color + "Player").classList.toggle(
      "current",
      next.turn === value && next.status !== "finished",
    );
    $(color + "TurnLabel").textContent =
      next.status === "finished"
        ? next.outcome === value
          ? "获胜"
          : next.outcome === 0
            ? "和棋"
            : "本局结束"
        : next.turn === value
          ? next.phase === "analysis"
            ? "局面分析中"
            : next.status === "running"
              ? "当前回合"
              : "等待开始"
          : "等待对手";
  }
  const humanTurn =
    next.status === "running" &&
    next.phase === "thinking" &&
    next.active_config?.human;
  cellButtons.forEach((button, index) => {
    const legal = next.legal.includes(index);
    button.disabled = !humanTurn || !legal;
    button.classList.toggle("legal", legal && humanTurn);
    button.setAttribute(
      "aria-label",
      `${coordinate(index)}，${next.cells[index] === 1 ? "黑棋" : next.cells[index] === -1 ? "白棋" : legal ? "合法空位" : "空位"}`,
    );
  });
  boardDirty = true;
  const side = next.turn === 1 ? "黑方" : "白方";
  let statusText = "准备就绪，黑方先行";
  if (next.status === "running")
    statusText =
      next.phase === "analysis"
        ? "统一分析局面 · 计时尚未开始"
        : humanTurn
          ? `${side}落子 · ${next.legal.length} 个合法落点`
          : `${side}思考中 · ${next.progress.strategy || ALGORITHMS[next.active_config?.algorithm] || ""}`;
  else if (next.status === "paused") statusText = "对局已暂停 · 点击继续";
  else if (next.status === "finished")
    statusText =
      next.outcome === 0
        ? "棋局结束 · 和棋"
        : `棋局结束 · ${next.outcome === 1 ? "黑方" : "白方"}获胜`;
  $("turnStatus").textContent = statusText;
  $("startBtn").disabled =
    !["idle", "paused"].includes(next.status) ||
    next.experiment.status === "running";
  $("startBtn").textContent =
    next.status === "paused"
      ? "▶ 继续对局"
      : next.status === "running"
        ? "对弈进行中"
        : next.status === "finished"
          ? "对局已结束"
          : "▶ 开始对局";
  $("pauseBtn").disabled = next.status !== "running" || next.pause_requested;
  $("pauseBtn").textContent = next.pause_requested ? "落子后暂停" : "暂停";
  $("resultOverlay").hidden = next.status !== "finished";
  if (next.status === "finished") {
    $("resultTitle").textContent =
      next.outcome === 0
        ? "势均力敌，和棋。"
        : `${next.outcome === 1 ? "黑方" : "白方"}获胜`;
    $("resultReason").textContent =
      `${next.counts.black} : ${next.counts.white} · ${next.reason}`;
  }
  renderAnalysis(next.analysis);
  renderSearch(next.progress);
  renderHistory(next.history);
  renderExperiment(next.experiment);
  if (next.error && next.error !== was?.error) toast(next.error);
}
function renderAnalysis(analysis) {
  const probabilities = analysis.probabilities;
  // 最大余数法使三个整数百分比合计恰好 100%，不改变原始导出概率。
  let rounded = null;
  if (probabilities) {
    const colors = ["black", "draw", "white"];
    rounded = Object.fromEntries(
      colors.map((color) => [color, Math.floor(probabilities[color] * 100)]),
    );
    const remainder =
      100 - Object.values(rounded).reduce((sum, value) => sum + value, 0);
    colors.sort(
      (a, b) =>
        probabilities[b] * 100 -
        rounded[b] -
        (probabilities[a] * 100 - rounded[a]),
    );
    for (let index = 0; index < remainder; index++)
      rounded[colors[index % 3]]++;
  }
  const labels = {
    pending: "更新中",
    calibrated: "估算",
    provisional: "待充分分析",
    solved: "残局已解",
    terminal: "实际结果",
    error: "分析异常",
  };
  $("analysisBadge").textContent = labels[analysis.status] || "待分析";
  for (const color of ["black", "draw", "white"]) {
    $(color + "Prob").textContent = probabilities ? `${rounded[color]}%` : "—";
    $(color + "Bar").style.width = probabilities
      ? `${probabilities[color] * 100}%`
      : color === "draw"
        ? "10%"
        : "45%";
  }
  $("analysisNote").textContent =
    analysis.label || "对局开始后，由统一分析器自动更新。";
  $("analysisDepth").textContent = analysis.depth
    ? `完成深度 ${analysis.depth} 层`
    : "统一局面分析";
  $("solvedText").hidden = analysis.status !== "solved";
  if (analysis.status === "solved")
    $("solvedText").textContent =
      `双方最佳应对：${analysis.outcome === 0 ? "和棋" : analysis.outcome === 1 ? "黑方必胜" : "白方必胜"}`;
  const key = `${state.revision}:${analysis.status}`;
  if (probabilities && !seenAnalyses.has(key)) {
    seenAnalyses.add(key);
    trends.push({
      ply: state.history.length,
      value: probabilities.black + probabilities.draw / 2,
    });
  }
  const maxPly = Math.max(10, ...trends.map((point) => point.ply));
  $("trendPath").setAttribute(
    "d",
    trends
      .map(
        (point, index) =>
          `${index ? "L" : "M"}${((point.ply / maxPly) * 238 + 1).toFixed(1)},${(60 - point.value * 54).toFixed(1)}`,
      )
      .join(" "),
  );
}
function renderSearch(progress) {
  const last = state.history.at(-1);
  const data = progress.strategy
    ? progress
    : last && last.algorithm !== "human" && last.algorithm !== "pass"
      ? last
      : {};
  $("searchIndicator").classList.toggle(
    "busy",
    state.status === "running" &&
      state.phase === "thinking" &&
      !state.active_config?.human,
  );
  $("strategyLabel").textContent =
    data.strategy ||
    (state.active_config?.human ? "人类回合 · 等待落子" : "等待搜索开始");
  $("searchDepth").textContent = data.depth || "—";
  $("searchNodes").textContent = data.nodes
    ? Intl.NumberFormat("en", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(data.nodes)
    : "—";
  $("searchIterations").textContent = data.iterations
    ? data.iterations.toLocaleString()
    : "—";
  $("searchTime").textContent =
    data.elapsed !== undefined
      ? `${(data.search_elapsed ?? data.elapsed).toFixed(2)}s`
      : "—";
  $("bestMove").textContent =
    data.move !== null && data.move !== undefined
      ? `${data.exact ? "已完整搜索 · " : ""}当前推荐 ${coordinate(data.move)}${data.iterations ? " · 按访问次数选择" : " · 按搜索评分选择"}`
      : "候选落点将在搜索时出现";
}
function renderHistory(history) {
  const key = `${history.length}:${state.revision}`;
  if (key === lastHistoryKey) return;
  lastHistoryKey = key;
  $("moveCount").textContent = history.length;
  if (!history.length) {
    $("moveList").innerHTML =
      '<p class="empty-state">每一步，都是新的可能。<br>开始对局后在这里回看落子。</p>';
    return;
  }
  $("moveList").replaceChildren();
  for (const move of history.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "move-row";
    const name =
      move.algorithm === "human"
        ? "玩家"
        : move.algorithm === "pass"
          ? "跳过"
          : ALGORITHMS[move.algorithm];
    row.title = `${name} · ${move.strategy || ""} · ${move.nodes || 0} 状态`;
    row.innerHTML = `<span><span class="move-number">${String(move.ply).padStart(2, "0")}</span><i class="disc-dot ${move.color === 1 ? "black" : "white"}"></i>${move.color === 1 ? "黑方" : "白方"}</span><strong>${move.coordinate}</strong><time>${move.elapsed.toFixed(2)}s</time>`;
    $("moveList").appendChild(row);
  }
}
function renderClocks(now) {
  if (!state) return;
  const delta =
    state.status === "running" && state.phase === "thinking"
      ? (now - stateReceivedAt) / 1000
      : 0;
  const elapsed = state.turn_elapsed + delta;
  $("turnTime").innerHTML = `${elapsed.toFixed(2)}<small>s</small>`;
  const black = state.totals.black + (state.turn === 1 ? delta : 0),
    white = state.totals.white + (state.turn === -1 ? delta : 0);
  $("blackTotal").innerHTML = `${black.toFixed(1)}<small>s</small>`;
  $("whiteTotal").innerHTML = `${white.toFixed(1)}<small>s</small>`;
  $("totalTime").innerHTML = `${(black + white).toFixed(1)}<small>s</small>`;
  const budget = state.active_config?.human
    ? 60
    : state.active_config?.seconds || 60;
  $("timeTrack").style.width = `${Math.min(100, (elapsed / budget) * 100)}%`;
  $("timeTrack").parentElement.classList.toggle("danger", elapsed > 55);
}
let clockFrame = 0;
function frame(now) {
  if (boardDirty || animations.size) drawBoard(now);
  if (now - clockFrame > 40) {
    renderClocks(now);
    clockFrame = now;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* 批量实验：有明确总局数与时间上限估计，并保留取消前已完成的棋局。 */
for (const [key, name] of Object.entries(ALGORITHMS)) {
  const label = document.createElement("label");
  label.innerHTML = `<input type="checkbox" value="${key}" ${["minimax", "alphabeta", "mcts", "hybrid_v3"].includes(key) ? "checked" : ""}> ${name}`;
  $("experimentAlgorithms").appendChild(label);
  label.querySelector("input").addEventListener("change", experimentEstimate);
}
for (const seconds of BUDGETS) {
  const button = document.createElement("button");
  button.className = "budget-chip" + (seconds === 0.2 ? " selected" : "");
  button.dataset.seconds = seconds;
  button.textContent = `${seconds}s`;
  button.addEventListener("click", () => {
    button.classList.toggle("selected");
    experimentEstimate();
  });
  $("experimentBudgets").appendChild(button);
}
function experimentConfig() {
  return {
    algorithms: [
      ...$("experimentAlgorithms").querySelectorAll("input:checked"),
    ].map((item) => item.value),
    budgets: [...$("experimentBudgets").querySelectorAll(".selected")].map(
      (item) => Number(item.dataset.seconds),
    ),
    pairs: Number($("experimentPairs").value),
    seed: Number($("experimentSeed").value),
  };
}
function experimentEstimate() {
  const config = experimentConfig(),
    combinations =
      (config.algorithms.length * (config.algorithms.length - 1)) / 2;
  const games = combinations * 2 * config.pairs * config.budgets.length;
  const seconds =
    combinations *
    2 *
    config.pairs *
    config.budgets.reduce((sum, n) => sum + n, 0) *
    54;
  $("experimentEstimate").textContent =
    `${games} 局对弈 · 搜索预算合计约 ${Math.ceil(seconds / 60)} 分钟，实际用时会受提前终局和调度开销影响。`;
}
$("experimentPairs").addEventListener("change", experimentEstimate);
$("runExperiment").addEventListener("click", () =>
  perform("experiment", experimentConfig()),
);
$("stopExperiment").addEventListener("click", () => perform("stop_experiment"));
experimentEstimate();
function renderExperiment(experiment) {
  // 页面刷新后恢复正在运行的实验配置，避免默认档位与实际任务不一致。
  if (!experimentConfigInitialized && experiment.config) {
    const config = experiment.config;
    $("experimentAlgorithms")
      .querySelectorAll("input")
      .forEach((input) => {
        input.checked = config.algorithms.includes(input.value);
      });
    $("experimentBudgets")
      .querySelectorAll("button")
      .forEach((button) => {
        button.classList.toggle(
          "selected",
          config.budgets.includes(Number(button.dataset.seconds)),
        );
      });
    $("experimentPairs").value = String(config.pairs);
    $("experimentSeed").value = String(config.seed);
    experimentConfigInitialized = true;
    experimentEstimate();
  }
  $("experimentAlgorithms")
    .querySelectorAll("input")
    .forEach((input) => {
      input.disabled = experiment.status === "running";
    });
  $("experimentBudgets")
    .querySelectorAll("button")
    .forEach((button) => {
      button.disabled = experiment.status === "running";
    });
  $("experimentPairs").disabled = experiment.status === "running";
  $("experimentSeed").disabled = experiment.status === "running";
  const statusNames = {
    idle: "尚未开始",
    running: "进行中",
    completed: "已完成",
    cancelled: "已停止",
    error: "异常停止",
  };
  $("experimentStatus").textContent =
    statusNames[experiment.status] || experiment.status;
  $("experimentCompleted").textContent = experiment.completed;
  $("experimentTotal").textContent = ` / ${experiment.total} 局`;
  $("experimentProgressBar").style.width =
    `${experiment.total ? (experiment.completed / experiment.total) * 100 : 0}%`;
  const current = experiment.current;
  $("experimentCurrent").textContent =
    current && experiment.status === "running"
      ? `${ALGORITHMS[current.black]} vs ${ALGORITHMS[current.white]} · ${current.seconds}s · 第 ${current.ply} 步`
      : experiment.status === "completed"
        ? "实验已结束，可导出统计表和完整棋谱。"
        : experiment.status === "cancelled"
          ? "已保留完成的对局，未完成棋局不计入结果。"
          : experiment.error || "统一时间预算 · 同一开局成对换色";
  $("runExperiment").disabled =
    experiment.status === "running" || state.status === "running";
  $("stopExperiment").disabled = experiment.status !== "running";
  const key = JSON.stringify(experiment.summary);
  if (key === lastExperimentKey) return;
  lastExperimentKey = key;
  renderComparisonChart(experiment.summary || []);
  $("experimentRows").innerHTML =
    (experiment.summary || [])
      .map(
        (row) =>
          `<tr><td>${ALGORITHMS[row.algorithm]}</td><td>${row.seconds}s</td><td>${row.wins} / ${row.draws} / ${row.losses}</td><td>${(row.win_rate * 100).toFixed(1)}%</td><td>${(row.score_rate * 100).toFixed(1)}%</td><td>${row.mean_seconds.toFixed(3)}s</td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="6" class="empty-state">实验数据将在每局结束后更新。</td></tr>';
}

// 时间档位按离散实验条件排列；曲线呈现实际得分率，不插值虚构中间实验。
function renderComparisonChart(rows) {
  $("comparisonChart").hidden = rows.length === 0;
  if (!rows.length) return;
  const budgets = [...new Set(rows.map((row) => row.seconds))].sort(
    (a, b) => a - b,
  );
  const colors = {
    minimax: "#a16e58",
    alphabeta: "#426b51",
    mcts: "#708ca9",
    hybrid: "#b28c4c",
    hybrid_v2: "#88709e",
    hybrid_v3: "#25877d",
  };
  const x = (budget) =>
    budgets.length === 1
      ? 300
      : 48 + (budgets.indexOf(budget) / (budgets.length - 1)) * 510;
  const y = (rate) => 174 - rate * 145;
  let svg = "";
  for (const rate of [0, 0.5, 1]) {
    svg += `<line x1="48" x2="558" y1="${y(rate)}" y2="${y(rate)}" stroke="#e1e6d9" stroke-dasharray="4 4"/><text x="36" y="${y(rate) + 4}" text-anchor="end" fill="#939e88" font-size="11">${rate * 100}%</text>`;
  }
  for (const budget of budgets)
    svg += `<text x="${x(budget)}" y="201" text-anchor="middle" fill="#939e88" font-size="11">${budget}s</text>`;
  const active = Object.keys(ALGORITHMS).filter((algorithm) =>
    rows.some((row) => row.algorithm === algorithm),
  );
  for (const algorithm of active) {
    const points = rows
      .filter((row) => row.algorithm === algorithm)
      .sort((a, b) => a.seconds - b.seconds);
    svg += `<polyline points="${points.map((row) => `${x(row.seconds)},${y(row.score_rate)}`).join(" ")}" fill="none" stroke="${colors[algorithm]}" stroke-width="2" opacity="0.8"/>`;
    for (const row of points)
      svg += `<circle cx="${x(row.seconds)}" cy="${y(row.score_rate)}" r="4" fill="${colors[algorithm]}" stroke="#fffefa" stroke-width="1"><title>${ALGORITHMS[algorithm]} · ${row.seconds}s · ${(row.score_rate * 100).toFixed(1)}% · ${row.games}局</title></circle>`;
  }
  $("comparisonSvg").innerHTML = svg;
  $("comparisonLegend").innerHTML = active
    .map(
      (algorithm) =>
        `<span><i style="background:${colors[algorithm]}"></i>${ALGORITHMS[algorithm]}</span>`,
    )
    .join("");
}

/* 展示验证数据，而不把概率误差误称为“准确率”。 */
async function loadCalibration() {
  try {
    const model = await (await fetch("/api/calibration")).json();
    const value = model.validation;
    if (!value) {
      $("calibrationDetails").textContent =
        "尚未生成校准模型。请运行 python3 tools/calibrate.py；在此之前不会显示未经校准的百分比。";
      return;
    }
    $("calibrationDetails").innerHTML =
      `<div class="calibration-grid"><div><span>训练 / 保留验证棋局</span><strong>${value.train_games} / ${value.heldout_games}</strong></div><div><span>验证 Brier 分数</span><strong>${value.brier_multiclass.toFixed(3)}</strong></div><div><span>类别先验基线 Brier</span><strong>${value.prior_baseline_brier.toFixed(3)}</strong></div><div><span>验证对数损失</span><strong>${value.log_loss.toFixed(3)}</strong></div></div><p class="muted">绿色：分组实际黑胜比例；铜色标记：平均预测概率。下方是保留验证集的分组结果，样本数包含黑白镜像。</p>`;
    for (let i = 0; i < value.reliability_bins.length; i++) {
      const bin = value.reliability_bins[i];
      if (!bin.count) continue;
      const row = document.createElement("div");
      row.className = "reliability-row";
      row.innerHTML = `<span>预测 ${i * 10}–${(i + 1) * 10}%</span><div class="reliability-bar"><i style="width:${bin.observed * 100}%"></i><b style="left:${bin.predicted * 100}%"></b></div><span>${bin.count} 局面</span>`;
      $("calibrationDetails").appendChild(row);
    }
  } catch (error) {
    $("calibrationDetails").textContent = "暂时无法读取校准记录。";
  }
}
$("refreshCalibration").addEventListener("click", loadCalibration);

/* 顺序轮询避免请求堆叠；页面后台时降低频率。 */
async function poll() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) throw new Error("服务暂时不可用");
    receive(await response.json());
  } catch (error) {
    $("connectionDot").style.background = "#bd7755";
    $("connectionText").textContent = "连接中断 · 请检查本地服务";
  }
  setTimeout(poll, document.hidden ? 1000 : 160);
}
poll();

/* 进化页只读取已保存的实验；筛选结果与独立验证分开展示。 */
async function loadEvolution() {
  try {
    const response = await fetch("/api/evolution");
    const data = await response.json();
    $("evolutionConclusion").textContent =
      data.conclusion || "独立验证进行中，完成后展示结果。";
    $("evolutionRows").innerHTML =
      (data.comparisons || [])
        .map((row) => {
          const interval = row.paired_bootstrap_95;
          return `<tr><td>${ALGORITHMS[row.candidate]}</td><td>${ALGORITHMS[row.opponent]}</td><td>${row.seconds}s</td><td>${row.wins} / ${row.draws} / ${row.losses}</td><td>${(row.win_rate * 100).toFixed(1)}%</td><td>${(row.score_rate * 100).toFixed(1)}%</td><td>${interval ? interval.map((n) => (n * 100).toFixed(1) + "%").join(" – ") : "—"}</td></tr>`;
        })
        .join("") || '<tr><td colspan="7">等待独立验证数据。</td></tr>';
  } catch (error) {
    $("evolutionConclusion").textContent =
      "暂时无法读取实验记录，请确认本地服务已启动。";
  }
}

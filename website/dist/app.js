"use strict";
const $ = (id) => document.getElementById(id);
const COLORS = [
  "#8c9a60",
  "#c47c58",
  "#718ca6",
  "#b7a25a",
  "#9c86ac",
  "#c28a98",
  "#5f998c",
  "#6d8061",
];
const PIECES = {
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
  K: "♚",
  Q: "♛",
  R: "♜",
  B: "♝",
  N: "♞",
  P: "♟",
};
const PIECE_NAMES = {
  k: "king",
  q: "queen",
  r: "rook",
  b: "bishop",
  n: "knight",
  p: "pawn",
};
let mode = "strategic";
let data,
  selected,
  ply = 0,
  enabled = new Set(),
  visible = [],
  view = [0, 0, 800, 570],
  drag;
let selectedId = location.hash.slice(1);
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const short = (name) => name.replace(/,/g, ", ");
const year = (game) => game.date.slice(0, 4);
function featureAverage(game, name) {
  return data.representation.featureNames.reduce(
    (sum, feature, i) =>
      sum + (feature.endsWith(": " + name) ? game.strategicFeatures[i] / 3 : 0),
    0,
  );
}
function color(game) {
  const by = $("color-by").value;
  if (by === "opening")
    return COLORS[Math.max(0, "ABCDE".indexOf(game.eco[0]))];
  if (by === "king" || by === "pawn") {
    const value = featureAverage(
      game,
      by === "king" ? "king zone attacked" : "pawn advancement",
    );
    return `hsl(${Math.round(95 - 85 * Math.min(1, value * 1.6))} 30% ${Math.round(73 - 30 * Math.min(1, value * 1.6))}%)`;
  }
  return COLORS[data.players.indexOf(game.player) % COLORS.length];
}
$("color-by").addEventListener("change", () => {
  const labels = {
    player: "Color identifies the reference player",
    opening: "Color identifies ECO family A–E",
    king: "Color shows target king exposure: pale = lower, dark = higher",
    pawn: "Color shows target pawn advancement: pale = lower, dark = higher",
  };
  document.querySelector(".map-footer > span").textContent =
    labels[$("color-by").value];
  renderMap();
  renderNeighbors();
});
const modeFields = {
  strategic: ["vector", "position"],
  baseline: ["baselineVector", "baselinePosition"],
  frozen: ["frozenVector", "frozenPosition"],
  temporal: ["temporalVector", "temporalPosition"],
  player: ["playerVector", "playerPosition"],
};
const vector = (g) => g[modeFields[mode][0]];
const position = (g) => g[modeFields[mode][1]];
function neighborCandidates() {
  return visible.filter(
    (g) =>
      g.id !== selected.id &&
      (!$("cross-opening").checked ||
        (selected.eco && g.eco && g.eco !== selected.eco)),
  );
}
$("cross-opening").addEventListener("change", () => {
  renderMap();
  renderNeighbors();
});
const distance = (a, b) =>
  Math.sqrt(vector(a).reduce((sum, v, i) => sum + (v - vector(b)[i]) ** 2, 0));
const blockLabels = {
  placement: "piece placement",
  pawns: "pawn structure",
  activity: "activity",
  king: "king safety",
  material: "material",
  decisions: "move choices",
};
function alignedBlocks(game) {
  if (mode === "baseline") return "Whole-game statistical baseline";
  if (mode === "player") return "Optimized explicitly for player identity";
  if (mode === "frozen") return "Frozen neural position-and-move match";
  if (mode === "temporal") return "Learned temporal embedding match";
  const groups = data.representation.blocks,
    scores = {};
  groups.forEach((group, i) => {
    scores[group] =
      (scores[group] || 0) +
      (game.vector[i] - selected.vector[i]) ** 2 /
        data.representation.blockWeights[group];
  });
  return (
    "Closest blocks: " +
    Object.entries(scores)
      .sort((a, b) => a[1] - b[1])
      .slice(0, 2)
      .map(([group]) => blockLabels[group])
      .join(", ")
  );
}
function updateMethod() {
  const r = data.representation;
  const retention = {
    strategic: r.neighborRetention10,
    baseline: r.baselineNeighborRetention10,
    frozen: r.frozenNeighborRetention10,
    temporal: r.temporalNeighborRetention10,
    player: r.playerNeighborRetention10,
  }[mode];
  const title = {
    strategic: "Strategic fingerprint",
    baseline: "Simple statistics",
    frozen: "Frozen Maia-3 + chosen moves",
    temporal: "Learned temporal encoder",
    player: "Player-supervised control",
  }[mode];
  $("variance").textContent =
    `${title}: the displayed map retains ${(retention * 100).toFixed(1)}% of each game's ten nearest neighbors on average. This measures projection fidelity. Retrieval uses full-dimensional vectors.`;
  document.querySelector(".axis-label").textContent =
    data.schemaVersion >= 3
      ? "t-SNE →"
      : mode === "strategic"
        ? "t-SNE →"
        : "PCA 1 →";
  document.querySelector(".similar-heading > span").textContent =
    `${vector(data.games[0]).length}-dimensional neighbors`;
  document.querySelector(".pill").textContent = {
    strategic: "STRATEGIC",
    baseline: "BASELINE",
    frozen: "PRETRAINED",
    temporal: "LEARNED",
    player: "PLAYER ID",
  }[mode];
}
$("representation").addEventListener("change", (e) => {
  mode = e.target.value;
  resetView();
  updateMethod();
  renderMap();
  renderNeighbors();
});
function announce(text) {
  $("announcement").textContent = text;
}
function openAbout() {
  $("about").showModal();
}
["about-side", "about-top", "method"].forEach((id) =>
  $(id).addEventListener("click", openAbout),
);
$("close-about").addEventListener("click", () => $("about").close());
$("about").addEventListener("click", (e) => {
  if (e.target === $("about")) {
    const r = $("about").getBoundingClientRect();
    if (
      e.clientX < r.left ||
      e.clientX > r.right ||
      e.clientY < r.top ||
      e.clientY > r.bottom
    )
      $("about").close();
  }
});

function renderPlayers() {
  $("players").replaceChildren();
  data.players.forEach((player, i) => {
    const button = document.createElement("button");
    button.className = "player-filter";
    button.setAttribute("aria-pressed", enabled.has(player));
    button.innerHTML = `<span class="color-dot" style="background:${COLORS[i]}"></span><span>${escapeHTML(player)}</span><span class="number">${data.games.filter((g) => g.player === player).length}</span>`;
    button.addEventListener("click", () => {
      enabled.has(player) ? enabled.delete(player) : enabled.add(player);
      renderPlayers();
      filter();
    });
    $("players").append(button);
  });
}
function filter() {
  if (!data) return;
  const query = $("search").value.trim().toLowerCase(),
    result = $("result").value;
  visible = data.games.filter(
    (g) =>
      enabled.has(g.player) &&
      (result === "all" || g.result === result) &&
      [g.player, g.white, g.black, g.event, g.eco, g.date]
        .join(" ")
        .toLowerCase()
        .includes(query),
  );
  $("visible-count").textContent = visible.length;
  $("empty").hidden = visible.length > 0;
  $("map-status").textContent =
    `${data.players.length} collections · ${data.games.length} unique games`;
  $("random").disabled = !visible.length;
  if (
    visible.length &&
    (!selected || !visible.some((g) => g.id === selected.id))
  )
    selectGame(visible[0]);
  else {
    renderMap();
    renderNeighbors();
  }
  announce(`${visible.length} games match your filters.`);
}
function resetFilters() {
  enabled = new Set(data.players);
  $("search").value = "";
  $("result").value = "all";
  renderPlayers();
  filter();
}
$("search").addEventListener("input", filter);
$("result").addEventListener("change", filter);
$("reset-players").addEventListener("click", () => {
  if (data) {
    enabled = new Set(data.players);
    renderPlayers();
    filter();
  }
});
$("clear").addEventListener("click", resetFilters);
$("random").addEventListener("click", () => {
  if (visible.length) {
    selectGame(visible[Math.floor(Math.random() * visible.length)]);
    resetView();
  }
});

function coordinates(game) {
  const xs = data.games.map((g) => position(g)[0]),
    ys = data.games.map((g) => position(g)[1]);
  const minx = Math.min(...xs),
    maxx = Math.max(...xs),
    miny = Math.min(...ys),
    maxy = Math.max(...ys);
  // Equal scaling preserves geometry; no independent stretching of PCA axes.
  const scale = Math.min(650 / (maxx - minx || 1), 420 / (maxy - miny || 1));
  return [
    400 + (position(game)[0] - (minx + maxx) / 2) * scale,
    285 - (position(game)[1] - (miny + maxy) / 2) * scale,
  ];
}
function svgEl(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attributes)) element.setAttribute(k, v);
  return element;
}
function renderMap() {
  if (!data) return;
  const map = $("map");
  map.replaceChildren();
  const near = selected
    ? neighborCandidates()
        .sort((a, b) => distance(a, selected) - distance(b, selected))
        .slice(0, 3)
    : [];
  if (selected && visible.some((g) => g.id === selected.id)) {
    const [x, y] = coordinates(selected);
    near.forEach((g) => {
      const [nx, ny] = coordinates(g);
      map.append(
        svgEl("line", {
          x1: x,
          y1: y,
          x2: nx,
          y2: ny,
          stroke: "#99a58d",
          "stroke-width": 0.8,
          "stroke-dasharray": "3 5",
          opacity: 0.65,
        }),
      );
    });
  }
  const ordered = [...visible].sort(
    (a, b) => Number(a.id === selected?.id) - Number(b.id === selected?.id),
  );
  ordered.forEach((g) => {
    const [x, y] = coordinates(g),
      isSelected = g.id === selected?.id;
    if (isSelected) {
      map.append(
        svgEl("circle", {
          cx: x,
          cy: y,
          r: 14,
          fill: color(g),
          "fill-opacity": 0.12,
          stroke: color(g),
          "stroke-width": 0.8,
        }),
      );
    }
    const dot = svgEl("circle", {
      cx: x,
      cy: y,
      r: isSelected ? 6 : 4.3,
      fill: color(g),
      opacity: isSelected ? 1 : 0.74,
      stroke: isSelected ? "#fcfcf8" : "#fcfcf8",
      "stroke-width": isSelected ? 2 : 1,
      class: "map-dot",
      tabindex: 0,
      role: "button",
      "aria-label": `${g.white} versus ${g.black}, ${year(g)}. ${isSelected ? "Selected." : ""}`,
    });
    dot.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!drag?.moved) selectGame(g);
    });
    dot.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectGame(g);
      }
    });
    dot.addEventListener("pointerenter", (e) => showTooltip(g, e));
    dot.addEventListener("pointerleave", () => {
      $("tooltip").hidden = true;
    });
    dot.addEventListener("focus", () => {
      const rect = dot.getBoundingClientRect();
      showTooltip(g, { clientX: rect.x, clientY: rect.y });
    });
    dot.addEventListener("blur", () => {
      $("tooltip").hidden = true;
    });
    map.append(dot);
  });
  map.setAttribute("viewBox", view.join(" "));
}
function showTooltip(g, e) {
  const wrap = $("map-wrap").getBoundingClientRect(),
    tip = $("tooltip");
  tip.innerHTML = `<strong>${escapeHTML(short(g.white))}</strong><br>vs ${escapeHTML(short(g.black))}<br><span>${escapeHTML(g.event)} · ${year(g)}</span>`;
  tip.hidden = false;
  tip.style.left = `${Math.min(Math.max(8, e.clientX - wrap.left + 12), Math.max(8, wrap.width - 240))}px`;
  tip.style.top = `${Math.max(8, Math.min(e.clientY - wrap.top + 12, wrap.height - 105))}px`;
}
function resetView() {
  view = [0, 0, 800, 570];
  $("map").setAttribute("viewBox", view.join(" "));
}
function zoom(factor) {
  const width = Math.min(1200, Math.max(160, view[2] * factor)),
    height = (width * 570) / 800;
  view = [
    view[0] + (view[2] - width) / 2,
    view[1] + (view[3] - height) / 2,
    width,
    height,
  ];
  $("map").setAttribute("viewBox", view.join(" "));
  $("tooltip").hidden = true;
}
$("zoom-in").addEventListener("click", () => zoom(0.8));
$("zoom-out").addEventListener("click", () => zoom(1.25));
$("reset-view").addEventListener("click", resetView);
$("map").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoom(e.deltaY > 0 ? 1.1 : 0.9);
  },
  { passive: false },
);
$("map").addEventListener("pointerdown", (e) => {
  if (e.target.tagName === "circle") return;
  drag = { x: e.clientX, y: e.clientY, view: [...view], moved: false };
  $("map").setPointerCapture(e.pointerId);
  $("map").classList.add("dragging");
});
$("map").addEventListener("pointermove", (e) => {
  if (!drag) return;
  const rect = $("map").getBoundingClientRect(),
    dx = e.clientX - drag.x,
    dy = e.clientY - drag.y;
  drag.moved = Math.abs(dx) + Math.abs(dy) > 3;
  view = [
    drag.view[0] - (dx * drag.view[2]) / rect.width,
    drag.view[1] - (dy * drag.view[3]) / rect.height,
    drag.view[2],
    drag.view[3],
  ];
  $("map").setAttribute("viewBox", view.join(" "));
});
function endDrag() {
  $("map").classList.remove("dragging");
  drag = null;
}
$("map").addEventListener("pointerup", endDrag);
$("map").addEventListener("pointercancel", endDrag);

let selectionRequest = 0;
const replayChunks = new Map();
async function selectGame(game, updateHash = true) {
  const request = ++selectionRequest;
  if (!game.moves && game.replayChunk) {
    $("move-label").textContent = "Loading replay…";
    for (const id of [
      "first",
      "prev",
      "next",
      "last",
      "download",
      "move-slider",
    ])
      $(id).disabled = true;
    try {
      if (!replayChunks.has(game.replayChunk))
        replayChunks.set(
          game.replayChunk,
          fetch(game.replayChunk)
            .then((response) => {
              if (!response.ok) throw new Error("Replay request failed");
              return response.json();
            })
            .catch((error) => {
              replayChunks.delete(game.replayChunk);
              throw error;
            }),
        );
      const chunk = await replayChunks.get(game.replayChunk);
      if (request !== selectionRequest) return;
      if (!chunk[game.id]) throw new Error("Replay missing from chunk");
      Object.assign(game, chunk[game.id]);
    } catch (error) {
      if (request === selectionRequest) {
        $("move-label").textContent =
          "Replay unavailable — select again to retry";
        announce("Replay could not be loaded. Select the game again to retry.");
      }
      return;
    }
  }
  $("download").disabled = false;
  $("move-slider").disabled = false;
  selected = game;
  ply = 0;
  if (updateHash) history.replaceState(null, "", `#${game.id}`);
  $("white-name").textContent = short(game.white);
  $("black-name").textContent = short(game.black);
  $("game-meta").textContent =
    `${game.event} · ${game.date.replaceAll(".", " / ")}`;
  $("game-result").textContent =
    game.result === "1/2-1/2"
      ? "½–½ · Draw"
      : game.result === "1-0"
        ? "1–0 · White wins"
        : game.result === "0-1"
          ? "0–1 · Black wins"
          : "Unfinished";
  $("game-eco").textContent = `ECO ${game.eco}`;
  $("game-length").textContent =
    `${Math.ceil((game.moves?.length ?? game.plies) / 2)} moves`;
  $("game-number").textContent =
    `NO. ${String(data.games.indexOf(game) + 1).padStart(3, "0")}`;
  $("move-slider").max = game.moves.length;
  $("game-split").textContent = game.split ? `${game.split} split` : "";
  $("move-list").replaceChildren();
  game.moves.forEach((move, i) => {
    if (i % 2 === 0) {
      const n = document.createElement("span");
      n.className = "move-num";
      n.textContent = `${i / 2 + 1}.`;
      $("move-list").append(n);
    }
    const button = document.createElement("button");
    button.textContent = move;
    button.dataset.ply = i + 1;
    button.setAttribute(
      "aria-label",
      `Move ${Math.floor(i / 2) + 1}, ${i % 2 ? "black" : "white"}: ${move}`,
    );
    button.addEventListener("click", () => setPly(i + 1));
    $("move-list").append(button);
  });
  const average = (name) =>
    data.representation.featureNames.reduce(
      (sum, feature, i) =>
        sum +
        (feature.endsWith(": " + name) ? game.strategicFeatures[i] / 3 : 0),
      0,
    );
  const featureInfo = [
    [
      "Center control",
      average("center control"),
      `${Math.round(average("center control") * 100)}%`,
    ],
    [
      "King exposure",
      average("king zone attacked"),
      `${Math.round(average("king zone attacked") * 100)}%`,
    ],
    [
      "Passed pawns",
      average("passed pawns"),
      (average("passed pawns") * 8).toFixed(1),
    ],
    [
      "Check rate",
      average("check rate"),
      `${Math.round(average("check rate") * 100)}%`,
    ],
  ];
  document.querySelector(".fingerprint .section-label").textContent =
    `${game.player.toUpperCase()} · POSITIONAL AVERAGES`;
  $("features").innerHTML = featureInfo
    .map(
      ([name, value, label]) =>
        `<div class="feature"><span>${name}</span><div class="feature-track"><div class="feature-fill" style="width:${Math.min(100, value * 100)}%"></div></div><span class="feature-value">${label}</span></div>`,
    )
    .join("");
  $("share").textContent = "Copy game link ↗";
  $("tooltip").hidden = true;
  renderBoard();
  renderMap();
  renderNeighbors();
  announce(`Selected ${game.white} versus ${game.black}, ${year(game)}.`);
}
function setPly(value) {
  if (!selected) return;
  ply = Math.max(0, Math.min(selected.moves.length, Number(value)));
  renderBoard();
}
function renderBoard() {
  const squares = selected.fens[ply]
    .split(" ")[0]
    .split("/")
    .flatMap((row) =>
      [...row].flatMap((char) =>
        /[1-8]/.test(char) ? Array(Number(char)).fill("") : char,
      ),
    );
  const board = $("board");
  board.replaceChildren();
  squares.forEach((piece, i) => {
    const rank = 8 - Math.floor(i / 8),
      file = "abcdefgh"[i % 8],
      square = document.createElement("div");
    square.className = `square ${(Math.floor(i / 8) + (i % 8)) % 2 ? "dark" : ""} ${piece ? (piece === piece.toUpperCase() ? "light-piece" : "dark-piece") : ""}`;
    square.textContent = PIECES[piece] || "";
    square.setAttribute("aria-hidden", "true");
    if (i % 8 === 0) {
      const coord = document.createElement("span");
      coord.className = "coord";
      coord.textContent = rank;
      square.append(coord);
    }
    if (rank === 1) {
      const coord = document.createElement("span");
      coord.className = "coord file";
      coord.textContent = file;
      square.append(coord);
    }
    board.append(square);
  });
  const description = squares
    .map((piece, i) =>
      piece
        ? `${piece === piece.toUpperCase() ? "white" : "black"} ${PIECE_NAMES[piece.toLowerCase()]} on ${"abcdefgh"[i % 8]}${8 - Math.floor(i / 8)}`
        : null,
    )
    .filter(Boolean)
    .join(", ");
  board.setAttribute(
    "aria-label",
    `Position after ${ply} plies: ${description}`,
  );
  $("move-label").textContent = ply
    ? `${Math.ceil(ply / 2)}${ply % 2 ? "." : "…"} ${selected.moves[ply - 1]}`
    : "Starting position";
  $("move-slider").value = ply;
  ["first", "prev"].forEach((id) => ($(id).disabled = ply === 0));
  ["next", "last"].forEach(
    (id) => ($(id).disabled = ply === selected.moves.length),
  );
  $("move-list")
    .querySelectorAll("button")
    .forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.ply) === ply);
      b.setAttribute(
        "aria-current",
        Number(b.dataset.ply) === ply ? "step" : "false",
      );
    });
  const active = $("move-list").querySelector(".active");
  if (active) {
    $("move-list").scrollTop = active.offsetTop - $("move-list").offsetTop - 20;
  } else $("move-list").scrollTop = 0;
}
function renderNeighbors() {
  $("neighbors").replaceChildren();
  if (!selected) return;
  const neighbors = neighborCandidates()
    .sort((a, b) => distance(a, selected) - distance(b, selected))
    .slice(0, 3);
  if (!neighbors.length) {
    $("neighbors").textContent =
      "Broaden your filters to discover neighboring games.";
    return;
  }
  neighbors.forEach((game) => {
    const b = document.createElement("button");
    b.className = "neighbor";
    b.innerHTML = `<span class="color-dot" style="background:${color(game)}"></span><span class="neighbor-name">${escapeHTML(game.white.split(",")[0])} – ${escapeHTML(game.black.split(",")[0])}</span><span class="neighbor-meta">${year(game)} · ${escapeHTML(game.eco)}<br>${Math.ceil((game.moves?.length ?? game.plies) / 2)} moves · ${escapeHTML(game.result)}</span><span class="distance">${distance(game, selected).toFixed(2)} distance ↗</span><span class="match-reason">${alignedBlocks(game)}</span>`;
    b.addEventListener("click", () => selectGame(game));
    $("neighbors").append(b);
  });
}
$("first").addEventListener("click", () => setPly(0));
$("prev").addEventListener("click", () => setPly(ply - 1));
$("next").addEventListener("click", () => setPly(ply + 1));
$("last").addEventListener("click", () => setPly(selected.moves.length));
$("move-slider").addEventListener("input", (e) => setPly(e.target.value));
$("download").addEventListener("click", () => {
  if (!selected) return;
  const link = document.createElement("a"),
    url = URL.createObjectURL(
      new Blob([selected.pgn], { type: "application/x-chess-pgn" }),
    );
  link.href = url;
  link.download = `game-atlas-${selected.id}.pgn`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("share").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    $("share").textContent = "Link copied ✓";
    announce("Game link copied.");
  } catch {
    $("share").textContent = "Copy the URL above";
    announce("Copy the current browser address to share this game.");
  }
});
window.addEventListener("hashchange", () => {
  const game = data?.games.find((g) => g.id === location.hash.slice(1));
  if (game) {
    resetFilters();
    selectGame(game, false);
  }
});
window.addEventListener("keydown", (e) => {
  if (
    $("about").open ||
    ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)
  )
    return;
  if (e.key === "ArrowRight") {
    e.preventDefault();
    setPly(ply + 1);
  }
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    setPly(ply - 1);
  }
});
function percent(value) {
  return value == null ? "—" : (value * 100).toFixed(1) + "%";
}
async function loadExperiments() {
  if (data.schemaVersion < 3) return;
  try {
    const response = await fetch(data.resultsFile || "experiments.json");
    if (!response.ok) throw new Error("Experiment report unavailable");
    const report = await response.json();
    if (report.display_games !== data.games.length)
      throw new Error("Report and atlas disagree");
    const labels = {
      strategic: "Strategic fingerprint",
      frozen_position: "Frozen positions",
      frozen_decision: "Frozen positions + moves",
      joint_7: "Temporal · seed 7",
      joint_21: "Temporal · seed 21",
      joint_42: "Temporal · seed 42",
      concept_42: "Concept-only temporal",
      player_42: "Player-supervised control",
    };
    $("results-summary").textContent =
      `${report.samples.toLocaleString()} game perspectives from ${report.unique_games.toLocaleString()} unique games. The comparison uses ${report.query_samples.toLocaleString()} held-out queries against ${report.reference_samples.toLocaleString()} training references. All perspectives of a game share one split.`;
    const facts = [
      [report.unique_games.toLocaleString(), "unique source games"],
      [report.split_samples.train.toLocaleString(), "training perspectives"],
      [report.display_games.toLocaleString(), "games to explore"],
      ["0", "canonical games shared across splits"],
    ];
    $("experiment-facts").innerHTML = facts
      .map(
        ([number, label]) =>
          `<div class="experiment-fact"><strong>${number}</strong><span>${label}</span></div>`,
      )
      .join("");
    $("experiment-rows").replaceChildren();
    for (const [key, result] of Object.entries(report.results)) {
      const row = document.createElement("tr");
      if (key === report.selected_temporal_run)
        row.className = "selected-model";
      const ci = result.standard.player_top1_ci95;
      row.innerHTML = `<td>${escapeHTML(labels[key] || key)}${key === report.selected_temporal_run ? " · shown on map" : ""}</td><td>${percent(result.standard.player_top1)}</td><td>${percent(ci[0])}–${percent(ci[1])}</td><td>${percent(result.cross_eco.player_top1)}</td><td>${percent(result.modern_cross_eco?.player_top1)}</td><td>${percent(result.standard.same_eco_top1)}</td>`;
      $("experiment-rows").append(row);
    }
    const seeds = Object.entries(report.results)
      .filter(([key]) => key.startsWith("joint_"))
      .map(([, r]) => r.standard.player_top1);
    const mean = seeds.reduce((a, b) => a + b, 0) / seeds.length;
    const sd = Math.sqrt(
      seeds.reduce((a, b) => a + (b - mean) ** 2, 0) /
        Math.max(1, seeds.length - 1),
    );
    const random = report.results.strategic.standard.random_player_top1;
    $("experiment-notes").innerHTML =
      `<div class="experiment-note"><strong>Repeatability across three seeds</strong>Temporal player retrieval: ${percent(mean)} mean, ${(sd * 100).toFixed(1)} percentage points standard deviation. The displayed model was selected by validation loss. Random-reference top-1 is about ${percent(random)}.</div><div class="experiment-note"><strong>Check the shortcuts</strong>A separate classifier using year, rating, ECO code, color and game length identifies players at ${percent(report.metadata_player_classifier_accuracy)} accuracy. It is a different task, but shows why player identification alone cannot establish style.</div>`;
    const comparison =
      report.comparisons_to_strategic?.[report.selected_temporal_run]?.standard;
    if (comparison) {
      const card = document.createElement("div");
      card.className = "experiment-note";
      const pp = (value) => (value >= 0 ? "+" : "") + (value * 100).toFixed(1);
      card.innerHTML = `<strong>Paired comparison on fresh held-out groups</strong>The displayed temporal model changes player top-1 by ${pp(comparison.mean_difference)} percentage points versus the strategic baseline; the grouped 95% interval is ${pp(comparison.ci95[0])} to ${pp(comparison.ci95[1])} points. An interval spanning zero does not establish a gain. The audit excludes ${report.benchmark.excluded_exploratory_game_groups.toLocaleString()} previously explored game groups.`;
      $("experiment-notes").append(card);
    }
  } catch (error) {
    console.error(error);
    $("results-summary").textContent =
      "The experiment report could not be loaded. Refresh to retry.";
  }
}

async function start() {
  try {
    const response = await fetch("atlas.json");
    if (!response.ok)
      throw new Error(`Dataset request returned ${response.status}`);
    data = await response.json();
    if (![2, 3].includes(data.schemaVersion) || !data.games.length)
      throw new Error("Unsupported or empty dataset");
    mode = data.schemaVersion >= 3 ? "temporal" : "strategic";
    $("representation").value = mode;
    for (const option of $("representation").options) {
      option.disabled = !data.games[0][modeFields[option.value][0]];
    }
    enabled = new Set(data.players);
    visible = [...data.games];
    renderPlayers();
    updateMethod();
    $("dataset-note").textContent =
      `${data.games.length} games · ${data.players.length} collections · Reproducible by design`;
    await selectGame(
      data.games.find((g) => g.id === selectedId) ||
        data.games.find((g) => g.player === "Tal") ||
        data.games[0],
      false,
    );
    filter();
    loadExperiments();
  } catch (error) {
    console.error(error);
    $("map-status").textContent = "The collection could not be loaded.";
    $("white-name").textContent = "Unable to load games";
    $("black-name").textContent = "Please refresh to try again.";
    $("random").disabled = true;
    announce("Unable to load the collection. Please refresh.");
  }
}
start();

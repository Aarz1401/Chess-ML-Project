/* Buildless position explorer. All retrieval and anonymous judgments stay local. */
"use strict";
if (new URLSearchParams(location.search).get("review") === "1")
  document.body.classList.add("review-only");
const $ = (id) => document.getElementById(id);
const PIECES = {
  K: "wK",
  Q: "wQ",
  R: "wR",
  B: "wB",
  N: "wN",
  P: "wP",
  k: "bK",
  q: "bQ",
  r: "bR",
  b: "bB",
  n: "bN",
  p: "bP",
};
const NAMES = {
  k: "king",
  q: "queen",
  r: "rook",
  b: "bishop",
  n: "knight",
  p: "pawn",
};
let dataset,
  review,
  report,
  queryIndex,
  matchIndex,
  neighbors = [],
  cursor = 0;
let responses = {},
  browserId = "",
  storageAvailable = true;
let frames = { source: null, match: null },
  steps = { source: 0, match: 0 },
  replayRequest = { source: 0, match: 0 };
const chunks = new Map();
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );

function squares(fen) {
  return fen
    .split(" ")[0]
    .split("/")
    .flatMap((row) =>
      [...row].flatMap((c) =>
        /[1-8]/.test(c) ? Array(Number(c)).fill("") : c,
      ),
    );
}
function positionKey(fen) {
  let [board, turn, castle, ep] = fen.split(" ");
  if (turn === "b") {
    board = board
      .split("/")
      .reverse()
      .join("/")
      .replace(/[a-zA-Z]/g, (c) =>
        c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase(),
      );
    castle = castle.replace(/[a-zA-Z]/g, (c) =>
      c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase(),
    );
    castle =
      "KQkq"
        .split("")
        .filter((c) => castle.includes(c))
        .join("") || "-";
    if (ep !== "-") ep = ep[0] + (9 - Number(ep[1]));
  }
  return [board, "w", castle, ep].join(" ");
}
function board(id, fen, orientation) {
  const cells = squares(fen),
    target = $(id);
  target.replaceChildren();
  const order = Array.from({ length: 64 }, (_, i) =>
    orientation === "b" ? 63 - i : i,
  );
  for (const i of order) {
    const piece = cells[i],
      square = document.createElement("div");
    square.className = `moment-square ${(Math.floor(i / 8) + (i % 8)) % 2 ? "dark" : ""} ${piece ? (piece === piece.toUpperCase() ? "white-piece" : "black-piece") : ""}`;
    if (piece) {
      const image = document.createElement("img");
      image.src = "assets/pieces/" + PIECES[piece] + ".svg";
      image.alt = "";
      image.className = "chess-piece";
      image.draggable = false;
      square.append(image);
    }
    const displayIndex = order.indexOf(i);
    if (displayIndex >= 56) {
      const c = document.createElement("small");
      c.textContent = "abcdefgh"[i % 8];
      square.append(c);
    }
    target.append(square);
  }
  const description = cells
    .map((p, i) =>
      p
        ? `${p === p.toUpperCase() ? "white" : "black"} ${NAMES[p.toLowerCase()]} on ${"abcdefgh"[i % 8]}${8 - Math.floor(i / 8)}`
        : null,
    )
    .filter(Boolean)
    .join(", ");
  target.setAttribute(
    "aria-label",
    `${orientation === "b" ? "Black" : "White"} at bottom. ${description}`,
  );
}
function eligible(q, c, cross = true) {
  const a = dataset.games[q.game],
    b = dataset.games[c.game];
  return (
    a.id !== b.id &&
    positionKey(q.fen) !== positionKey(c.fen) &&
    b.split === "train" &&
    (!cross || Boolean(a.eco && b.eco && a.eco !== b.eco))
  );
}
function rank(q, method, cross = true) {
  const v = dataset.vectors[method],
    from = v[q];
  return dataset.moments
    .map((m, i) => ({
      index: i,
      distance: eligible(dataset.moments[q], m, cross)
        ? Math.sqrt(v[i].reduce((s, x, j) => s + (x - from[j]) ** 2, 0))
        : Infinity,
    }))
    .filter((m) => Number.isFinite(m.distance))
    .sort((a, b) => a.distance - b.distance || a.index - b.index)
    .filter(
      (m, i, all) =>
        all.findIndex(
          (t) =>
            dataset.moments[t.index].game === dataset.moments[m.index].game,
        ) === i,
    )
    .slice(0, 5);
}
function turnText(moment) {
  return `${moment.fen.split(" ")[1] === "w" ? "White" : "Black"} to move · move ${Math.floor(moment.ply / 2) + 1} · side to move at bottom`;
}
function title(game) {
  return `${game.white} — ${game.black}`;
}
function balance(value) {
  return value > 0 ? `+${value}` : String(value);
}
function engineText(moment) {
  const e = moment.engine,
    score =
      e.mate != null ? `mate ${e.mate}` : `${(e.cp / 100).toFixed(2)} pawns`;
  return `Starting moment · Stockfish ${score} for the original side to move. Top move: ${e.bestSan}. Fixed ${report.engine.nodes_limit.toLocaleString()}-node search (depth ${e.depth}); approximate, not a verified plan.`;
}
async function json(file) {
  const r = await fetch(file);
  if (!r.ok) throw new Error(`${file}: ${r.status}`);
  return r.json();
}
async function loadFrames(which, index) {
  const token = ++replayRequest[which],
    m = dataset.moments[index],
    g = dataset.games[m.game];
  frames[which] = null;
  steps[which] = 0;
  $(`${which}-prev`).disabled = true;
  $(`${which}-next`).disabled = true;
  try {
    if (!chunks.has(g.replayChunk))
      chunks.set(
        g.replayChunk,
        json(g.replayChunk).catch((e) => {
          chunks.delete(g.replayChunk);
          throw e;
        }),
      );
    const chunk = await chunks.get(g.replayChunk);
    if (token !== replayRequest[which]) return;
    const replay = chunk[g.id];
    if (!replay || replay.fens[m.ply] !== m.fen)
      throw new Error("Replay and sampled position disagree");
    frames[which] = {
      fens: replay.fens.slice(m.ply, m.ply + 13),
      moves: replay.moves.slice(m.ply, m.ply + 12),
    };
    renderFrame(which, index);
  } catch (e) {
    if (token === replayRequest[which])
      $(`${which}-step`).textContent =
        "Continuation unavailable · refresh to retry";
    console.error(e);
  }
}
function renderFrame(which, index) {
  const m = dataset.moments[index],
    step = steps[which],
    replay = frames[which];
  board(`${which}-board`, replay?.fens[step] || m.fen, m.fen.split(" ")[1]);
  $(`${which}-step`).textContent = step
    ? `Historical continuation +${step} plies: ${replay.moves[step - 1]}`
    : "Starting moment";
  $(`${which}-prev`).disabled = step === 0;
  $(`${which}-next`).disabled = !replay || step >= replay.fens.length - 1;
}
function renderPosition(which, index) {
  const m = dataset.moments[index],
    g = dataset.games[m.game];
  $(`${which}-meta`).textContent =
    `${title(g)} · ${g.date.slice(0, 4)} · ${g.eco || "ECO unknown"} · ${g.split}`;
  $(`${which}-turn`).textContent = turnText(m);
  $(`${which}-eval`).textContent = engineText(m);
  $(`${which}-game-link`).href = `index.html#${g.id}`;
  board(`${which}-board`, m.fen, m.fen.split(" ")[1]);
  loadFrames(which, index);
}
function comparisons(a, b) {
  const rows = [];
  if (a.isolated && b.isolated)
    rows.push([
      "Isolated pawns in both",
      `Source: ${a.isolated}. Match: ${b.isolated}. Neither position has an adjacent-file friendly pawn supporting those pawns.`,
    ]);
  if (a.passed && b.passed)
    rows.push([
      "Passed pawns in both",
      `Source: ${a.passed}. Match: ${b.passed}. There are no opposing pawns ahead on the same or adjacent files.`,
    ]);
  if (a.kingWing === b.kingWing && a.enemyKingWing === b.enemyKingWing)
    rows.push([
      "Similar king placement",
      `Target king: ${a.kingWing}. Opposing king: ${a.enemyKingWing}, in both positions. This describes location, not castling history.`,
    ]);
  rows.push([
    "Material balance",
    `Source ${balance(a.material)} · match ${balance(b.material)} pawn units, from each target side's perspective.`,
  ]);
  rows.push([
    "Pawn structure",
    `Isolated ${a.isolated} / ${b.isolated} · passed ${a.passed} / ${b.passed} · doubled ${a.doubled} / ${b.doubled} (source / match).`,
  ]);
  rows.push([
    "King safety signals",
    `Attacked king-zone squares ${a.kingAttacked}/${a.kingZone} vs ${b.kingAttacked}/${b.kingZone}. Nearby forward shield pawns ${a.shield} vs ${b.shield}. These counts do not establish an attack.`,
  ]);
  rows.push([
    "Central activity",
    `The target side controls ${a.center}/4 vs ${b.center}/4 central squares. Queens remaining: ${a.queens} vs ${b.queens}.`,
  ]);
  return rows.slice(0, 6);
}
function chooseMatch(index) {
  matchIndex = index;
  renderPosition("match", index);
  $("moment-reasons").innerHTML = comparisons(
    dataset.moments[queryIndex].facts,
    dataset.moments[index].facts,
  )
    .map(
      ([h, t]) =>
        `<div class="fact-comparison"><strong>${escapeHTML(h)}</strong><span>${escapeHTML(t)}</span></div>`,
    )
    .join("");
  document
    .querySelectorAll(".moment-neighbor")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        Number(b.dataset.index) === index ? "true" : "false",
      ),
    );
}
function searchMatches() {
  neighbors = rank(
    queryIndex,
    $("moment-method").value,
    $("moment-cross").checked,
  );
  $("moment-neighbors").replaceChildren();
  for (const item of neighbors) {
    const m = dataset.moments[item.index],
      g = dataset.games[m.game],
      b = document.createElement("button");
    b.className = "moment-neighbor";
    b.dataset.index = item.index;
    b.innerHTML = `${escapeHTML(title(g))}<span>Move ${Math.floor(m.ply / 2) + 1} · ${escapeHTML(g.eco)} · distance ${item.distance.toFixed(3)}</span>`;
    b.addEventListener("click", () => chooseMatch(item.index));
    $("moment-neighbors").append(b);
  }
  if (neighbors.length) chooseMatch(neighbors[0].index);
  else {
    matchIndex = null;
    ++replayRequest.match;
    frames.match = null;
    $("match-board").replaceChildren();
    $("match-meta").textContent =
      "No eligible match. Try allowing the same ECO code.";
    for (const id of [
      "match-turn",
      "match-eval",
      "match-step",
      "moment-reasons",
    ])
      $(id).textContent = "";
    $("match-prev").disabled = true;
    $("match-next").disabled = true;
    $("match-game-link").removeAttribute("href");
  }
}
function selectQuery(index, updateHash = true) {
  queryIndex = index;
  const m = dataset.moments[index];
  $("moment-game").value = String(m.game);
  $("moment-ply").replaceChildren();
  dataset.moments.forEach((p, i) => {
    if (p.game === m.game) {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `Move ${Math.floor(p.ply / 2) + 1} · ${p.fen.split(" ")[1] === "w" ? "White" : "Black"} to move`;
      $("moment-ply").append(o);
    }
  });
  $("moment-ply").value = index;
  if (updateHash) history.replaceState(null, "", `#${m.id}`);
  renderPosition("source", index);
  searchMatches();
}
function storageKey() {
  return `game-atlas-review-${dataset.version}`;
}
function saveResponses() {
  try {
    localStorage.setItem(
      storageKey(),
      JSON.stringify({ browserId, responses }),
    );
  } catch {
    storageAvailable = false;
  }
}
function trials() {
  return review.trials.filter(
    (t) => t.partition === $("review-partition").value,
  );
}
function renderReview() {
  const list = trials(),
    t = list[cursor];
  for (const [id, i] of [
    ["review-query", t.query],
    ["review-a", t.A],
    ["review-b", t.B],
  ]) {
    const m = dataset.moments[i];
    board(id, m.fen, m.fen.split(" ")[1]);
    $(`${id}-turn`).textContent = turnText(m);
  }
  const completed = list.filter((t) => responses[t.id]).length;
  $("review-progress").textContent =
    `Comparison ${cursor + 1}/${list.length} · ${completed} recorded${storageAvailable ? "" : " · storage unavailable; export before leaving"}`;
  $("review-back").disabled = cursor === 0;
  $("review-skip").disabled = cursor === list.length - 1;
  $("audit-warning").hidden = $("review-partition").value !== "audit";
  document
    .querySelectorAll("[data-judgment]")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        responses[t.id]?.choice === b.dataset.judgment ? "true" : "false",
      ),
    );
}
function judge(choice) {
  const list = trials(),
    t = list[cursor];
  responses[t.id] = {
    trialId: t.id,
    choice,
    partition: t.partition,
    experience: $("review-experience").value,
    recordedAt: new Date().toISOString(),
  };
  saveResponses();
  if (cursor < list.length - 1) cursor++;
  renderReview();
}
function download(value, name) {
  const link = document.createElement("a"),
    url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
    );
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function renderEvidence() {
  $("moment-facts").innerHTML = [
    [report.moments.toLocaleString(), "sampled positions"],
    [report.games.toLocaleString(), "replayable games"],
    [report.fit_positions.toLocaleString(), "train-only fitting positions"],
    ["100", "prepared blind comparisons"],
  ]
    .map(
      ([n, l]) =>
        `<div class="experiment-fact"><strong>${n}</strong><span>${l}</span></div>`,
    )
    .join("");
  const b = report.benchmark;
  $("moment-evidence").textContent =
    `${b.query_games} held-out query games against ${b.reference_positions.toLocaleString()} training positions, with different exact ECO codes and exact transpositions excluded. The two methods choose different top matches for ${(100 * b.disagreement_fraction).toFixed(1)}% of queries.`;
  $("moment-diagnostics").innerHTML = Object.entries(b.diagnostics)
    .map(
      ([name, r]) =>
        `<tr><td>${{ strategic: "Strategic position", neural: "Frozen neural position", random: "Random eligible position" }[name]}</td><td>${(r.median_engine_gap_cp / 100).toFixed(2)} pawns</td><td>${r.median_material_gap_pawns.toFixed(1)} pawn units</td><td>${(100 * r.same_player_fraction).toFixed(1)}%</td></tr>`,
    )
    .join("");
  $("moment-human-status").textContent =
    `Human relevance results: pending. ${report.human_evaluation.responses} independently collected judgments are included in this release. Engine: ${report.engine.id.name}, ${report.engine.nodes_limit.toLocaleString()} nodes per position, median depth ${report.engine.median_depth}.`;
}
async function start() {
  try {
    [dataset, review, report] = await Promise.all([
      json("moments.json"),
      json("moment-review.json"),
      json("moment-report.json"),
    ]);
    if (
      dataset.version !== review.version ||
      dataset.version !== report.version ||
      dataset.schemaVersion !== 1
    )
      throw new Error("Moment data versions disagree");
    $("moment-game").replaceChildren();
    dataset.games.forEach((g, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `${title(g)} · ${g.date.slice(0, 4)}`;
      $("moment-game").append(o);
    });
    document.querySelectorAll("button,select").forEach((control) => {
      control.disabled = false;
    });
    $("moment-status").textContent =
      `${dataset.moments.length.toLocaleString()} positions · ${dataset.games.length.toLocaleString()} games · choose a moment, compare the match, then follow the moves.`;
    browserId = crypto.randomUUID();
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey()) || "null");
      if (saved) {
        browserId = saved.browserId || browserId;
        const ids = new Set(review.trials.map((t) => t.id));
        responses = Object.fromEntries(
          Object.entries(saved.responses || {}).filter(
            ([id, r]) =>
              ids.has(id) && ["A", "B", "tie", "neither"].includes(r.choice),
          ),
        );
      }
    } catch {
      storageAvailable = false;
    }
    const wanted = decodeURIComponent(location.hash.slice(1));
    const selected = dataset.moments.findIndex(
      (m) => m.id === wanted || dataset.games[m.game].id === wanted,
    );
    selectQuery(selected >= 0 ? selected : review.trials[0].query, false);
    renderReview();
    renderEvidence();
  } catch (e) {
    console.error(e);
    $("moment-status").textContent =
      "The position collection could not be loaded. Refresh to retry.";
    document
      .querySelectorAll("button,select")
      .forEach((b) => (b.disabled = true));
  }
}
$("moment-game").addEventListener("change", () =>
  selectQuery(
    dataset.moments.findIndex((m) => m.game === Number($("moment-game").value)),
  ),
);
$("moment-ply").addEventListener("change", () =>
  selectQuery(Number($("moment-ply").value)),
);
$("moment-method").addEventListener("change", searchMatches);
$("moment-cross").addEventListener("change", searchMatches);
$("moment-surprise").addEventListener("click", () =>
  selectQuery(Math.floor(Math.random() * dataset.moments.length)),
);
for (const which of ["source", "match"])
  for (const [action, delta] of [
    ["prev", -1],
    ["next", 1],
  ])
    $(`${which}-${action}`).addEventListener("click", () => {
      const f = frames[which];
      if (!f) return;
      steps[which] = Math.max(
        0,
        Math.min(f.fens.length - 1, steps[which] + delta),
      );
      renderFrame(which, which === "source" ? queryIndex : matchIndex);
    });
document
  .querySelectorAll("[data-judgment]")
  .forEach((b) => b.addEventListener("click", () => judge(b.dataset.judgment)));
$("review-partition").addEventListener("change", () => {
  cursor = 0;
  renderReview();
});
$("review-back").addEventListener("click", () => {
  cursor = Math.max(0, cursor - 1);
  renderReview();
});
$("review-skip").addEventListener("click", () => {
  cursor = Math.min(trials().length - 1, cursor + 1);
  renderReview();
});
$("review-export").addEventListener("click", () =>
  download(
    {
      schemaVersion: 1,
      version: dataset.version,
      browserId,
      exportedAt: new Date().toISOString(),
      responses: Object.values(responses),
    },
    `atlas-judgments-${dataset.version}.json`,
  ),
);
$("review-clear").addEventListener("click", () => {
  if (
    confirm(
      "Clear all judgments stored in this browser for this dataset? Export first if you want to keep them.",
    )
  ) {
    responses = {};
    saveResponses();
    cursor = 0;
    renderReview();
  }
});
window.addEventListener("hashchange", () => {
  if (!dataset) return;
  const index = dataset.moments.findIndex(
    (m) => m.id === decodeURIComponent(location.hash.slice(1)),
  );
  if (index >= 0) selectQuery(index, false);
});
window.MomentMatching = { positionKey, rank, eligible, comparisons };
document.querySelectorAll("button,select").forEach((control) => {
  control.disabled = true;
});
start();

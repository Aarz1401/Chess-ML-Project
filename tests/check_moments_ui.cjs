/* Real exported assets in JSDOM; no human judgments are written to project data. */
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path");
const { createRequire } = require("node:module");
const { JSDOM } = createRequire(
  path.resolve(__dirname, "../website/package.json"),
)("jsdom");
const directory = path.resolve(__dirname, "../website/dist");
const data = JSON.parse(fs.readFileSync(path.join(directory, "moments.json")));
const review = JSON.parse(
  fs.readFileSync(path.join(directory, "moment-review.json")),
);
const dom = new JSDOM(
  fs.readFileSync(path.join(directory, "moments.html"), "utf8"),
  {
    url: "http://atlas.test/Chess-ML-Project/moments.html",
    runScripts: "outside-only",
  },
);
const w = dom.window,
  $ = (id) => w.document.getElementById(id),
  errors = [];
w.console.error = (...args) => errors.push(args.map(String).join(" "));
w.fetch = async (name) => {
  const file = path.join(
    directory,
    new URL(
      String(name),
      "http://atlas.test/Chess-ML-Project/",
    ).pathname.replace(/^\/Chess-ML-Project\//, ""),
  );
  return {
    ok: fs.existsSync(file),
    status: fs.existsSync(file) ? 200 : 404,
    json: async () => JSON.parse(fs.readFileSync(file, "utf8")),
  };
};
w.confirm = () => true;
w.URL.createObjectURL = (blob) => {
  w.__exportBlob = blob;
  return "blob:fixture";
};
w.URL.revokeObjectURL = () => {};
w.HTMLAnchorElement.prototype.click = function () {
  w.__filename = this.download;
};
w.eval(fs.readFileSync(path.join(directory, "moments.js"), "utf8"));
const wait = async () => new Promise((r) => setTimeout(r, 100));
(async () => {
  await wait();
  assert.deepEqual(errors, []);
  assert.equal(
    w.document.querySelectorAll("#source-board .moment-square").length,
    64,
  );
  assert.equal(
    w.document.querySelectorAll("#match-board .moment-square").length,
    64,
  );
  assert.equal($("moment-game").options.length, data.games.length);
  assert.match(
    w.document.querySelector('#source-board img[src$="wK.svg"]').src,
    /\/Chess-ML-Project\/assets\/pieces\/wK\.svg$/,
  );
  assert.match(
    $("moment-human-status").textContent,
    /0 independently collected/,
  );
  assert.ok(w.document.querySelectorAll(".moment-neighbor").length > 0);
  const q = review.trials[0].query;
  const actual = w.MomentMatching.rank(q, "strategic", true);
  assert.ok([review.trials[0].A, review.trials[0].B].includes(actual[0].index));
  for (const r of actual) {
    assert.notEqual(data.moments[q].game, data.moments[r.index].game);
    assert.notEqual(
      data.games[data.moments[q].game].eco,
      data.games[data.moments[r.index].game].eco,
    );
    assert.equal(data.games[data.moments[r.index].game].split, "train");
  }
  $("source-next").click();
  assert.match($("source-step").textContent, /Historical continuation/);
  $("source-prev").click();
  assert.equal($("source-step").textContent, "Starting moment");
  $("moment-method").value = "neural";
  $("moment-method").dispatchEvent(new w.Event("change"));
  await wait();
  assert.deepEqual(errors, []);
  assert.match($("moment-reasons").textContent, /Material balance/);
  $("moment-cross").checked = false;
  $("moment-cross").dispatchEvent(new w.Event("change"));
  await wait();
  const reviewText = w.document.querySelector(".review-boards").textContent;
  assert.doesNotMatch(reviewText, /Strategic|Maia|Stockfish/);
  w.document.querySelector('[data-judgment="neither"]').click();
  const saved = JSON.parse(
    w.localStorage.getItem(`game-atlas-review-${data.version}`),
  );
  assert.equal(Object.values(saved.responses)[0].choice, "neither");
  assert.match($("review-progress").textContent, /1 recorded/);
  $("review-partition").value = "audit";
  $("review-partition").dispatchEvent(new w.Event("change"));
  assert.equal($("audit-warning").hidden, false);
  assert.match($("review-progress").textContent, /1\/80/);
  $("review-export").click();
  assert.match(w.__filename, /^atlas-judgments-.*\.json$/);
  const exported = await new Promise((resolve, reject) => {
    const reader = new w.FileReader();
    reader.onload = () => resolve(JSON.parse(reader.result));
    reader.onerror = reject;
    reader.readAsText(w.__exportBlob);
  });
  assert.equal(exported.version, data.version);
  assert.equal(exported.responses.length, 1);
  assert.equal(exported.responses[0].choice, "neither");
  assert.equal(exported.responses[0].partition, "development");
  $("review-clear").click();
  assert.match($("review-progress").textContent, /0 recorded/);
  // A source selection change must also replace the continuation's replay.
  $("moment-game").value = "1";
  $("moment-game").dispatchEvent(new w.Event("change"));
  await wait();
  assert.ok(w.location.hash.includes(":"));
  assert.deepEqual(errors, []);
  console.log(
    "PASS: moment retrieval constraints, paired boards, continuations, method switch, explanations, blind choices, audit separation, local storage and export.",
  );
  dom.window.close();
})().catch((error) => {
  console.error(error);
  dom.window.close();
  process.exitCode = 1;
});

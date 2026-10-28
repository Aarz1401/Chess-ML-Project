/* DOM unit checks; these do not replace browser layout or accessibility testing. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { createRequire } = require("node:module");
const requireWebsite = createRequire(
  path.resolve(__dirname, "../website/package.json"),
);
const { JSDOM } = requireWebsite("jsdom");
const directory = path.resolve(__dirname, "../website/dist");
const dataset = JSON.parse(fs.readFileSync(path.join(directory, "atlas.json")));
const dom = new JSDOM(
  fs.readFileSync(path.join(directory, "index.html"), "utf8"),
  {
    url: "http://atlas.test/",
    runScripts: "outside-only",
  },
);
const window = dom.window;
const errors = [];
window.console.error = (...args) => errors.push(args.map(String).join(" "));
window.fetch = async (name) => {
  const file = path.join(
    directory,
    new URL(String(name), "http://atlas.test/").pathname.slice(1),
  );
  return {
    ok: fs.existsSync(file),
    status: fs.existsSync(file) ? 200 : 404,
    json: async () => JSON.parse(fs.readFileSync(file, "utf8")),
  };
};
window.navigator.clipboard = {
  writeText: async (text) => {
    window.__clipboard = text;
  },
};
window.URL.createObjectURL = () => "blob:test";
window.URL.revokeObjectURL = () => {};
window.HTMLAnchorElement.prototype.click = function () {
  window.__download = this.download;
};
window.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
window.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
window.eval(fs.readFileSync(path.join(directory, "app.js"), "utf8"));
const $ = (id) => window.document.getElementById(id);
(async () => {
  await new Promise((resolve) => setTimeout(resolve, 100));
  assert.deepEqual(errors, []);
  assert.equal(Number($("visible-count").textContent), dataset.games.length);
  assert.equal(
    window.document.querySelectorAll(".map-dot").length,
    dataset.games.length,
  );
  assert.equal(window.document.querySelectorAll("#board .square").length, 64);
  $("next").click();
  assert.equal($("move-slider").value, "1");
  assert.match($("board").getAttribute("aria-label"), /after 1 plies/);
  $("prev").click();
  assert.equal($("move-slider").value, "0");
  $("last").click();
  assert.equal($("move-slider").value, $("move-slider").max);
  $("first").click();
  assert.equal($("move-slider").value, "0");
  $("search").value = "no-such-game-zzzz";
  $("search").dispatchEvent(new window.Event("input"));
  assert.equal($("visible-count").textContent, "0");
  assert.equal($("empty").hidden, false);
  $("clear").click();
  assert.equal(Number($("visible-count").textContent), dataset.games.length);
  $("representation").value =
    dataset.schemaVersion >= 3 ? "frozen" : "baseline";
  $("representation").dispatchEvent(new window.Event("change"));
  assert.match(
    window.document.querySelector(".pill").textContent,
    /PRETRAINED|BASELINE/,
  );
  const filter = window.document.querySelector(".player-filter");
  filter.click();
  assert.ok(Number($("visible-count").textContent) < dataset.games.length);
  $("reset-players").click();
  assert.equal(Number($("visible-count").textContent), dataset.games.length);
  assert.equal(
    $("representation").value,
    dataset.schemaVersion >= 3 ? "frozen" : "baseline",
  );
  $("cross-opening").checked = true;
  $("cross-opening").dispatchEvent(new window.Event("change"));
  assert.ok(window.document.querySelectorAll(".neighbor").length <= 3);
  window.document
    .querySelector(".map-dot")
    .dispatchEvent(new window.Event("click"));
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.ok(window.location.hash.length > 1);
  $("share").click();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(window.__clipboard, /#[a-f0-9]+$/);
  $("download").click();
  assert.match(window.__download, /^game-atlas-[a-f0-9]+\.pgn$/);
  $("about-top").click();
  assert.equal($("about").open, true);
  $("close-about").click();
  assert.equal($("about").open, false);
  if (dataset.schemaVersion >= 3) {
    assert.ok($("experiment-rows").children.length >= 3);
    $("representation").value = "player";
    $("representation").dispatchEvent(new window.Event("change"));
    assert.equal(
      window.document.querySelector(".pill").textContent,
      "PLAYER ID",
    );
    assert.match($("results-summary").textContent, /held-out queries/);
  }
  assert.deepEqual(errors, []);
  console.log(
    "PASS: data loading, map, replay, search, filters, representation switch, sharing, PGN export, methodology and measured-report rendering.",
  );
  dom.window.close();
})().catch((error) => {
  console.error(error);
  dom.window.close();
  process.exitCode = 1;
});

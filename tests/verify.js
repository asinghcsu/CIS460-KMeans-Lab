// Compares kmeans.js with scikit-learn and with the numbers quoted in the course materials.
//   node tests/verify.js
const fs = require("fs");
const path = require("path");
const KM = require("../kmeans.js");
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));

let passed = 0, failed = 0;
function check(name, ok, detail = "") {
  if (ok) passed += 1;
  else { failed += 1; console.log(`FAIL ${name} ${detail}`); }
}
const near = (name, a, b, tol = 1e-3) => check(name, Math.abs(a - b) <= tol, `${a} vs ${b}`);
const sameArray = (name, a, b) => check(name, JSON.stringify(a) === JSON.stringify(b), `${a} vs ${b}`);

const scores = [58, 30, 59, 32, 61, 86, 42, 90, 92, 45];
const scoreOnly = scores.map((s) => [s]);
const twoD = scores.map((s, i) => [i + 1, s]);
const scaler = KM.standardize(twoD);
const starts = { good: [1, 4, 8], poor: [6, 5, 7] };

// Figures quoted on the page and in the deck.
const good = KM.run(scoreOnly, starts.good.map((i) => scoreOnly[i]));
near("deck: center 1", good.centers[0][0], 37.25);
near("deck: center 2", good.centers[1][0], 59.333, 1e-2);
near("deck: center 3", good.centers[2][0], 89.333, 1e-2);
near("deck: inertia", good.inertia, 186.08, 5e-3);
near("deck: poor inertia", KM.run(scoreOnly, starts.poor.map((i) => scoreOnly[i])).inertia, 1005.43, 5e-3);
near("deck: raw B-G", KM.dist(twoD[1], twoD[6]), 13.0);
near("deck: raw B-A", KM.dist(twoD[1], twoD[0]), 28.02, 5e-3);
near("deck: scaled B-G", KM.dist(scaler.X[1], scaler.X[6]), 1.82, 5e-3);
near("deck: scaled B-A", KM.dist(scaler.X[1], scaler.X[0]), 1.32, 5e-3);

// Scaler matches StandardScaler.
ref.scaler.mean.forEach((v, f) => near(`scaler mean ${f}`, scaler.mean[f], v));
ref.scaler.sd.forEach((v, f) => near(`scaler sd ${f}`, scaler.sd[f], v));
ref.two_d_scaled.X.forEach((row, i) => row.forEach((v, f) => near(`scaled ${i},${f}`, scaler.X[i][f], v)));

// Fixed-start runs match sklearn: same centers, same labels, same inertia.
[["score_only", scoreOnly], ["two_d_raw", twoD], ["two_d_scaled", scaler.X]].forEach(([name, X]) => {
  ["good", "poor"].forEach((kind) => {
    const r = KM.run(X, starts[kind].map((i) => X[i]));
    const e = ref[name][kind];
    near(`${name}/${kind} inertia`, r.inertia, e.inertia);
    sameArray(`${name}/${kind} labels`, r.labels, e.labels);
    r.centers.forEach((c, j) => c.forEach((v, f) => near(`${name}/${kind} c${j}`, v, e.centers[j][f])));
  });
  // The Choosing K table.
  Object.entries(ref[name].by_k).forEach(([k, e]) => {
    const r = KM.bestOf(X, Number(k), 50, KM.mulberry32(460));
    near(`${name} K=${k} inertia`, r.inertia, e.inertia);
    if (e.silhouette !== undefined) near(`${name} K=${k} silhouette`, KM.silhouette(X, r.labels), e.silhouette);
  });
});

// Moons.
const m = ref.moons;
near("moons ARI of sklearn labels", KM.ari(m.y, m.kmeans_labels), m.kmeans_ari);
const best = KM.bestOf(m.X, 2, 20, KM.mulberry32(460));
near("moons inertia", best.inertia, m.kmeans_inertia);
near("moons ARI", KM.ari(m.y, best.labels), m.kmeans_ari, 2e-2);

// Inertia never rises through an assign-then-update cycle, from any start.
const rng = KM.mulberry32(1);
let rose = 0;
for (let t = 0; t < 200; t += 1) {
  let centers = KM.randomInit(m.X.length, 4, rng).map((i) => m.X[i]);
  let labels = KM.assign(m.X, centers), last = KM.inertia(m.X, centers, labels);
  for (let i = 0; i < 30; i += 1) {
    centers = KM.means(m.X, labels, centers);
    const afterUpdate = KM.inertia(m.X, centers, labels);
    labels = KM.assign(m.X, centers);
    const afterAssign = KM.inertia(m.X, centers, labels);
    if (afterUpdate > last + 1e-9 || afterAssign > afterUpdate + 1e-9) rose += 1;
    last = afterAssign;
  }
}
check("inertia is non-increasing", rose === 0, `${rose} rises`);

// Starting-point helpers always return K distinct points.
[2, 3, 4].forEach((k) => {
  [scoreOnly, twoD, m.X].forEach((X, d) => {
    [KM.spreadInit(X, k), KM.poorInit(X, k), KM.randomInit(X.length, k, rng), KM.kppInit(X, k, rng)].forEach((idx, s) => {
      check(`init ${s} data ${d} K=${k} distinct`, new Set(idx).size === k, idx);
    });
  });
});

// k-means++ worked example on the page: first center B (30).
const seed = KM.seedState(scoreOnly, [1]);
near("kpp: total D2", seed.total, 13539);
near("kpp: D2 of I", seed.d2[8], 3844);
near("kpp: D2 of D", seed.d2[3], 4);
near("kpp: chance of I", seed.probs[8], 0.2839, 5e-5);
near("kpp: chance of D", seed.probs[3], 0.0003, 5e-5);
near("kpp: probabilities sum to 1", seed.probs.reduce((a, b) => a + b, 0), 1, 1e-12);
check("kpp: chosen point cannot repeat", seed.probs[1] === 0);

// Draws occur at the stated rates.
const counts = new Array(10).fill(0), drawRng = KM.mulberry32(7), draws = 200000;
for (let i = 0; i < draws; i += 1) counts[KM.pickSeed(seed.probs, drawRng())] += 1;
counts.forEach((c, i) => near(`kpp: draw rate ${i}`, c / draws, seed.probs[i], 5e-3));

// Identical points leave total D2 at 0; selection must still return unchosen points.
const same = [[1], [1], [1], [1]];
check("kpp: identical points stay distinct", new Set(KM.kppInit(same, 3, KM.mulberry32(2))).size === 3);

// DBSCAN matches scikit-learn exactly: labels, core points and ARI.
const db = ref.dbscan;
function dbCheck(name, X, c, truth) {
  const r = KM.dbscan(X, c.eps, c.min_samples);
  sameArray(`${name} eps=${c.eps} labels`, r.labels, c.labels);
  sameArray(`${name} eps=${c.eps} core`, r.core.flatMap((v, i) => (v ? [i] : [])), c.core);
  if (truth) near(`${name} eps=${c.eps} ari`, KM.ari(truth, r.labels), c.ari);
}
db.moons.forEach((c) => dbCheck("moons", m.X, c, m.y));
db.blobs.cases.forEach((c) => dbCheck("blobs", db.blobs.X, c, db.blobs.y));
db.score.forEach((c) => dbCheck("score", scoreOnly, c));

// Figures quoted on the page.
near("moons DBSCAN ARI on page", db.moons[0].ari, 1.0);
near("blobs K-Means ARI on page", db.blobs.kmeans_ari, 0.90, 5e-3);
near("blobs DBSCAN best ARI on page", db.blobs.grid_best_ari, 0.67, 5e-3);
check("blobs grid size on page", db.blobs.grid_size === 1121, db.blobs.grid_size);
check("blobs suggested settings", db.blobs.grid_best_eps === 0.85 && db.blobs.grid_best_min_samples === 11);
const page = KM.dbscan(scoreOnly, 5, 3);
const members = (g) => "ABCDEFGHIJ".split("").filter((_, i) => page.labels[i] === g).join("");
sameArray("page example: clusters", [members(0), members(1), members(-1)], ["ACE", "FHI", "BDGJ"]);
sameArray("page example: core points", "ABCDEFGHIJ".split("").filter((_, i) => page.core[i]).join(""), "ACEH");

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

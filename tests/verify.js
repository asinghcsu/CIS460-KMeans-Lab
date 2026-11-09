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

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

// K-Means engine with no page code, so tests/verify.js can compare it with scikit-learn.
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KM = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const sq = (a, b) => a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0);
  const dist = (a, b) => Math.sqrt(sq(a, b));

  // Seeded generator, so the Choosing K table is the same on every load.
  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Ties go to the lowest-numbered center, as numpy.argmin does.
  function assign(X, centers) {
    return X.map((p) => {
      let best = 0, bestD = Infinity;
      centers.forEach((c, j) => {
        const d = sq(p, c);
        if (d < bestD) { bestD = d; best = j; }
      });
      return best;
    });
  }

  // A center with no points stays where it is.
  function means(X, labels, centers) {
    return centers.map((c, j) => {
      const members = X.filter((_, i) => labels[i] === j);
      if (!members.length) return c.slice();
      return c.map((_, f) => members.reduce((s, p) => s + p[f], 0) / members.length);
    });
  }

  const inertia = (X, centers, labels) => X.reduce((s, p, i) => s + sq(p, centers[labels[i]]), 0);

  function run(X, startCenters, maxIter = 300) {
    let centers = startCenters.map((c) => c.slice());
    let labels = assign(X, centers);
    let iterations = 1;
    while (iterations < maxIter) {
      centers = means(X, labels, centers);
      const next = assign(X, centers);
      const same = next.every((v, i) => v === labels[i]);
      labels = next;
      if (same) break;
      iterations += 1;
    }
    return { centers, labels, inertia: inertia(X, centers, labels), iterations };
  }

  // Population standard deviation, as scikit-learn's StandardScaler uses.
  function standardize(X) {
    const n = X.length;
    const mean = X[0].map((_, f) => X.reduce((s, p) => s + p[f], 0) / n);
    const sd = X[0].map((_, f) => Math.sqrt(X.reduce((s, p) => s + (p[f] - mean[f]) ** 2, 0) / n) || 1);
    return { X: X.map((p) => p.map((v, f) => (v - mean[f]) / sd[f])), mean, sd };
  }

  function randomInit(n, k, rng) {
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i += 1) {
      const j = i + Math.floor(rng() * (n - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
  }

  function kppInit(X, k, rng) {
    const chosen = [Math.floor(rng() * X.length)];
    while (chosen.length < k) {
      const weights = X.map((p) => Math.min(...chosen.map((i) => sq(p, X[i]))));
      const total = weights.reduce((a, b) => a + b, 0);
      if (total === 0) {
        chosen.push(X.findIndex((_, i) => !chosen.includes(i)));
        continue;
      }
      let target = rng() * total, pick = weights.findIndex((w) => w > 0);
      for (let i = 0; i < X.length; i += 1) {
        target -= weights[i];
        if (weights[i] > 0) pick = i;
        if (target <= 0 && weights[i] > 0) break;
      }
      chosen.push(pick);
    }
    return chosen;
  }

  // Points at even steps along the first feature.
  function spreadInit(X, k) {
    const order = X.map((p, i) => [p[0], i]).sort((a, b) => a[0] - b[0]).map((e) => e[1]);
    return Array.from({ length: k }, (_, j) => order[Math.round((j * (X.length - 1)) / Math.max(k - 1, 1))]);
  }

  // The k points packed closest around the rightmost point.
  function poorInit(X, k) {
    let anchor = 0;
    X.forEach((p, i) => { if (p[0] > X[anchor][0]) anchor = i; });
    return X.map((p, i) => [sq(p, X[anchor]), i]).sort((a, b) => a[0] - b[0]).slice(0, k).map((e) => e[1]);
  }

  function bestOf(X, k, restarts, rng) {
    let best = null;
    for (let r = 0; r < restarts; r += 1) {
      const result = run(X, kppInit(X, k, rng).map((i) => X[i]));
      if (!best || result.inertia < best.inertia) best = result;
    }
    return best;
  }

  function silhouette(X, labels) {
    const groups = [...new Set(labels)];
    if (groups.length < 2) return null;
    const scores = X.map((p, i) => {
      const own = X.filter((_, j) => labels[j] === labels[i] && j !== i);
      if (!own.length) return 0;
      const a = own.reduce((s, q) => s + dist(p, q), 0) / own.length;
      let b = Infinity;
      groups.filter((g) => g !== labels[i]).forEach((g) => {
        const other = X.filter((_, j) => labels[j] === g);
        b = Math.min(b, other.reduce((s, q) => s + dist(p, q), 0) / other.length);
      });
      return (b - a) / Math.max(a, b);
    });
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  }

  function ari(truth, pred) {
    const pairs = (n) => (n * (n - 1)) / 2;
    const table = new Map(), rows = new Map(), cols = new Map();
    truth.forEach((t, i) => {
      const key = `${t}|${pred[i]}`;
      table.set(key, (table.get(key) || 0) + 1);
      rows.set(t, (rows.get(t) || 0) + 1);
      cols.set(pred[i], (cols.get(pred[i]) || 0) + 1);
    });
    const sum = (m) => [...m.values()].reduce((s, n) => s + pairs(n), 0);
    const index = sum(table), a = sum(rows), b = sum(cols);
    const expected = (a * b) / pairs(truth.length);
    const max = (a + b) / 2;
    return max === expected ? 1 : (index - expected) / (max - expected);
  }

  return { dist, assign, means, inertia, run, standardize, randomInit, kppInit, spreadInit, poorInit, bestOf, silhouette, ari, mulberry32 };
});

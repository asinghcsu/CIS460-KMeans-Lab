// K-Means and DBSCAN side by side on the same data.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const { el, shapePath, starPoints, niceTicks, extent, stackLevels, fmt, swatch } = window.LabDraw;
  const stacked = window.matchMedia("(max-width: 900px)");
  const LETTERS = "ABCDEFGHIJ";
  const SCORES = [58, 30, 59, 32, 61, 86, 42, 90, 92, 45];
  const COLORED = 4; // Only four cluster colors pass the color-vision check; further clusters share a neutral gray.

  const pointIds = (n) => Array.from({ length: n }, (_, i) => `P${i + 1}`);
  const SETS = {
    moons: { X: window.KM_MOONS.X, truth: window.KM_MOONS.y, ids: pointIds(200), features: ["Feature 1", "Feature 2"], k: 2, eps: 0.3, min: 5, range: [0.05, 0.6, 0.01] },
    blobs: { X: window.KM_BLOBS.X, truth: window.KM_BLOBS.y, ids: pointIds(200), features: ["Feature 1", "Feature 2"], k: 3, eps: 0.85, min: 11, range: [0.1, 3, 0.05] },
    score: { X: SCORES.map((s) => [s]), truth: null, ids: LETTERS.split(""), features: ["Exam score"], k: 3, eps: 5, min: 3, range: [1, 30, 1], levels: stackLevels(SCORES.map((s) => [s])) },
  };

  const cmp = { set: "moons", k: 2, eps: 0.3, min: 5, selected: 0 };
  const kmCache = new Map();

  function kmeansResult() {
    const key = `${cmp.set}|${cmp.k}`;
    if (!kmCache.has(key)) kmCache.set(key, KM.bestOf(SETS[cmp.set].X, cmp.k, 10, KM.mulberry32(460)));
    return kmCache.get(key);
  }

  // Gives each K-Means cluster the color of the DBSCAN cluster it overlaps most, so the panels can be compared.
  function matchColors(kmLabels, dbLabels, k) {
    const overlap = [];
    for (let j = 0; j < k; j += 1) {
      for (let slot = 0; slot < COLORED; slot += 1) {
        overlap.push([kmLabels.filter((v, i) => v === j && dbLabels[i] === slot).length, j, slot]);
      }
    }
    overlap.sort((a, b) => b[0] - a[0] || a[1] - b[1] || a[2] - b[2]);
    const map = {}, used = new Set();
    overlap.forEach(([n, j, slot]) => {
      if (n > 0 && map[j] === undefined && !used.has(slot)) { map[j] = slot; used.add(slot); }
    });
    for (let j = 0; j < k; j += 1) {
      if (map[j] === undefined) {
        let slot = 0;
        while (used.has(slot)) slot += 1;
        map[j] = slot;
        used.add(slot);
      }
    }
    return map;
  }

  function frame(svg, d) {
    svg.replaceChildren();
    const narrow = window.matchMedia("(max-width: 620px)").matches;
    const W = stacked.matches && narrow ? 360 : 540;
    const H = d.levels ? 250 : narrow ? 320 : 370;
    const m = { l: 44, r: 12, t: 12, b: d.levels ? 46 : 42 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const pw = W - m.l - m.r, ph = H - m.t - m.b, base = m.t + ph;
    const xDom = extent(d.X.map((p) => p[0]));
    const sx = (v) => m.l + ((v - xDom[0]) / (xDom[1] - xDom[0])) * pw;
    niceTicks(xDom[0], xDom[1], narrow ? 5 : 7).forEach((t) => {
      if (!d.levels) svg.appendChild(el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: base, class: "grid-line" }));
      svg.appendChild(el("text", { x: sx(t), y: base + (d.levels ? 26 : 16), "text-anchor": "middle", class: "tick" }, String(t)));
    });
    svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: base, y2: base, class: "axis-line" }));
    svg.appendChild(el("text", { x: m.l + pw / 2, y: H - 3, "text-anchor": "middle", class: "axis-title" }, d.features[0]));

    let sy, unitsY = 1;
    if (d.levels) {
      const rise = Math.min(46, (ph - 26) / (Math.max(...d.levels) + 1));
      sy = (i) => base - rise * (d.levels[i] + 1) - 6;
    } else {
      const yDom = extent(d.X.map((p) => p[1]));
      const f = (v) => m.t + ph - ((v - yDom[0]) / (yDom[1] - yDom[0])) * ph;
      unitsY = ph / (yDom[1] - yDom[0]);
      niceTicks(yDom[0], yDom[1], 5).forEach((t) => {
        svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: f(t), y2: f(t), class: "grid-line" }));
        svg.appendChild(el("text", { x: m.l - 6, y: f(t) + 4, "text-anchor": "end", class: "tick" }, String(t)));
      });
      svg.appendChild(el("line", { x1: m.l, x2: m.l, y1: m.t, y2: base, class: "axis-line" }));
      svg.appendChild(el("text", { x: 11, y: m.t + ph / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 11 ${m.t + ph / 2})` }, d.features[1]));
      sy = (i) => f(d.X[i][1]);
      sy.value = f;
    }
    return { W, base, m, ph, sx, sy, unitsX: pw / (xDom[1] - xDom[0]), unitsY };
  }

  function addPoint(svg, g, d, i, x, y, r, selected, drawMark) {
    const labelled = d.X.length <= 30;
    const group = el("g", { class: g.cls, "data-i": i, "data-tip": g.tip });
    if (labelled) {
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
      group.setAttribute("aria-label", g.tip);
    }
    group.appendChild(el("circle", { cx: x, cy: y, r: labelled ? 15 : 7, class: "hit" }));
    drawMark(group);
    if (selected) group.appendChild(el("circle", { cx: x, cy: y, r: r + 5, class: "select-ring" }));
    group.appendChild(el("circle", { cx: x, cy: y, r: r + 8, class: "select-ring-focus" }));
    svg.appendChild(group);
    if (labelled) svg.appendChild(el("text", { x: x + 8, y: y - 8, class: "pt-label" }, d.ids[i]));
  }

  function drawKMeans(d, km, colors) {
    const svg = $("cmpKmeans");
    const f = frame(svg, d);
    const r = d.X.length <= 30 ? 6.5 : 3.6;
    d.X.forEach((p, i) => {
      const slot = colors[km.labels[i]];
      const x = f.sx(p[0]), y = f.sy(i);
      addPoint(svg, { cls: `cmp-pt c${slot}`, tip: `${d.ids[i]}: K-Means cluster ${slot + 1}` }, d, i, x, y, r, i === cmp.selected,
        (g) => g.appendChild(el("path", { d: shapePath(slot, x, y, r), class: "mark" })));
    });
    km.centers.forEach((c, j) => {
      const x = f.sx(c[0]), y = d.levels ? f.base : f.sy.value(c[1]);
      svg.appendChild(el("polygon", { points: starPoints(x, y, 13, 6), class: `star c${colors[j]}`, "data-tip": `Center of cluster ${colors[j] + 1}` }));
    });
  }

  function drawDbscan(d, db) {
    const svg = $("cmpDbscan");
    const f = frame(svg, d);
    const r = d.X.length <= 30 ? 6.5 : 3.6;
    const sel = d.X[cmp.selected];
    // The eps neighborhood is an ellipse when the axes use different scales.
    if (d.levels) {
      const x1 = f.sx(sel[0] - cmp.eps), x2 = f.sx(sel[0] + cmp.eps);
      const lo = Math.max(f.m.l, x1), hi = Math.min(f.W - f.m.r, x2);
      svg.appendChild(el("rect", { x: lo, y: f.m.t, width: Math.max(hi - lo, 0), height: f.ph, class: "eps-area" }));
    } else {
      svg.appendChild(el("ellipse", { cx: f.sx(sel[0]), cy: f.sy(cmp.selected), rx: cmp.eps * f.unitsX, ry: cmp.eps * f.unitsY, class: "eps-area" }));
    }
    d.X.forEach((p, i) => {
      const label = db.labels[i];
      const x = f.sx(p[0]), y = f.sy(i);
      const kind = label < 0 ? "noise" : db.core[i] ? "core" : "border";
      const cls = label < 0 ? "cmp-pt noise" : `cmp-pt ${label < COLORED ? `c${label}` : "c-other"}${kind === "border" ? " hollow" : ""}`;
      const tip = `${d.ids[i]}: ${label < 0 ? "noise" : `cluster ${label + 1}, ${kind} point`}`;
      addPoint(svg, { cls, tip }, d, i, x, y, r, i === cmp.selected, (g) => {
        if (label < 0) {
          const s = r * 0.8;
          g.appendChild(el("path", { d: `M${x - s},${y - s}L${x + s},${y + s}M${x - s},${y + s}L${x + s},${y - s}`, class: "noise-mark" }));
        } else {
          g.appendChild(el("path", { d: shapePath(label < COLORED ? label : 0, x, y, r), class: "mark" }));
        }
      });
    });
  }

  function clusterSizes(labels) {
    const counts = {};
    labels.forEach((v) => { if (v >= 0) counts[v] = (counts[v] || 0) + 1; });
    const list = Object.values(counts).sort((a, b) => b - a);
    return list.length ? list.join(", ") : "None";
  }

  function members(d, labels, count, order) {
    const groups = [];
    for (let j = 0; j < count; j += 1) groups.push(`{${d.ids.filter((_, i) => labels[i] === order(j)).join(", ")}}`);
    return groups.join(" ");
  }

  function renderSummary(d, km, db) {
    const noise = db.labels.filter((v) => v < 0).length;
    const rows = [
      ["Clusters found", String(cmp.k), String(db.clusters)],
      ["Points left as noise", "0", String(noise)],
      ["Cluster sizes", clusterSizes(km.labels), clusterSizes(db.labels)],
    ];
    if (d.truth) rows.push(["ARI against the true groups", fmt(KM.ari(d.truth, km.labels)), fmt(KM.ari(d.truth, db.labels))]);
    if (d.X.length <= 30) {
      const noiseNames = d.ids.filter((_, i) => db.labels[i] < 0).join(", ");
      rows.push(["Clusters", members(d, km.labels, cmp.k, (j) => j), `${members(d, db.labels, db.clusters, (j) => j)}${noiseNames ? `; noise: ${noiseNames}` : ""}`]);
    }
    $("cmpTable").tBodies[0].innerHTML = rows.map(([a, b, c]) => `<tr><th scope="row">${a}</th><td class="wrap">${b}</td><td class="wrap">${c}</td></tr>`).join("");
    $("cmpKmeansHead").textContent = `K = ${cmp.k}`;
    $("cmpDbscanHead").textContent = `eps = ${cmp.eps}, min_samples = ${cmp.min}`;
    const hasOther = db.clusters > COLORED;
    $("cmpLegend").innerHTML = [
      `<span>${swatch(0)} Same shape and color: matching clusters</span>`,
      `<span><svg width="16" height="16" viewBox="0 0 30 30" aria-hidden="true"><polygon points="${starPoints(15, 15, 14, 6.5)}" fill="#fff" stroke="#15201b" stroke-width="2"/></svg> K-Means center</span>`,
      '<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="#2a78d6"/></svg> Core point</span>',
      '<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="4.5" fill="#fcfcfb" stroke="#2a78d6" stroke-width="2"/></svg> Border point</span>',
      '<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3,3L11,11M3,11L11,3" stroke="#6b736e" stroke-width="2"/></svg> Noise</span>',
      hasOther ? '<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="#8d938e"/></svg> Cluster 5 and above</span>' : "",
    ].join("");
  }

  function renderInspector(d, km, db, colors) {
    const i = cmp.selected;
    const name = d.ids[i];
    const n = db.neighbors[i].length;
    const label = db.labels[i];
    let dbText;
    if (db.core[i]) {
      dbText = `It has ${n} point${n === 1 ? "" : "s"} within eps, counting itself. That meets min_samples = ${cmp.min}, so it is a <strong>core point</strong> in cluster ${label + 1}.`;
    } else if (label >= 0) {
      const via = db.neighbors[i].find((j) => db.core[j] && db.labels[j] === label);
      dbText = `It has only ${n} point${n === 1 ? "" : "s"} within eps, fewer than min_samples = ${cmp.min}, but it lies within eps of core point ${d.ids[via]}. It is a <strong>border point</strong> in cluster ${label + 1}.`;
    } else {
      dbText = `It has only ${n} point${n === 1 ? "" : "s"} within eps, fewer than min_samples = ${cmp.min}, and no core point is within eps of it. It is <strong>noise</strong>.`;
    }
    $("cmpInspect").innerHTML = `<h3>Point ${name}: ${d.X[i].map(fmt).join(", ")}</h3><p><strong>K-Means.</strong> It joins cluster ${colors[km.labels[i]] + 1}, because its nearest center is that cluster's. K-Means assigns every point, however isolated.</p><p><strong>DBSCAN.</strong> ${dbText}</p>`;
  }

  function render() {
    const d = SETS[cmp.set];
    const km = kmeansResult();
    const db = KM.dbscan(d.X, cmp.eps, cmp.min);
    const colors = matchColors(km.labels, db.labels, cmp.k);
    drawKMeans(d, km, colors);
    drawDbscan(d, db);
    renderSummary(d, km, db);
    renderInspector(d, km, db, colors);
  }

  function syncControls() {
    const d = SETS[cmp.set];
    const eps = $("cmpEps");
    [eps.min, eps.max, eps.step] = d.range.map(String);
    eps.value = String(cmp.eps);
    $("cmpMin").value = String(cmp.min);
    $("cmpData").value = cmp.set;
    $("cmpK").value = String(cmp.k);
    $("cmpEpsValue").textContent = String(cmp.eps);
    $("cmpMinValue").textContent = String(cmp.min);
  }

  function load(set) {
    const d = SETS[set];
    Object.assign(cmp, { set, k: d.k, eps: d.eps, min: d.min, selected: 0 });
    syncControls();
    render();
  }

  $("cmpData").addEventListener("change", (e) => load(e.target.value));
  $("cmpK").addEventListener("change", (e) => { cmp.k = Number(e.target.value); render(); });
  $("cmpEps").addEventListener("input", (e) => { cmp.eps = Number(e.target.value); $("cmpEpsValue").textContent = e.target.value; render(); });
  $("cmpMin").addEventListener("input", (e) => { cmp.min = Number(e.target.value); $("cmpMinValue").textContent = e.target.value; render(); });
  $("cmpRestore").addEventListener("click", () => load(cmp.set));

  ["cmpKmeans", "cmpDbscan"].forEach((id) => {
    const svg = $(id);
    svg.addEventListener("click", (event) => {
      const point = event.target.closest(".cmp-pt");
      if (point) { cmp.selected = Number(point.dataset.i); render(); }
    });
    svg.addEventListener("keydown", (event) => {
      const point = event.target.closest(".cmp-pt");
      if (point && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        event.stopPropagation();
        const i = point.dataset.i;
        cmp.selected = Number(i);
        render();
        const again = svg.querySelector(`.cmp-pt[data-i="${i}"]`);
        if (again) again.focus();
      }
    });
  });
  stacked.addEventListener("change", render);
  window.matchMedia("(max-width: 620px)").addEventListener("change", render);

  window.LabCompare = { load };
  load("moons");
})();

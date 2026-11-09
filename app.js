(() => {
  "use strict";

  const svgNS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const fmt = (v) => Number(v).toFixed(2);
  const compactQuery = window.matchMedia("(max-width: 620px)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const HOURS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const SCORES = [58, 30, 59, 32, 61, 86, 42, 90, 92, 45];
  const STUDENT_STARTS = { good: [1, 4, 8], poor: [6, 5, 7] }; // B E I (30, 61, 92) and G F H (42, 86, 90)
  const DATA_NAMES = { score: "Students: score", hours: "Students: hours and score", moons: "Two moons", blobs: "Random groups" };
  const START_NAMES = { good: "Well spread", poor: "Poorly placed", random: "Random points", kpp: "k-means++" };
  const MAX_TABLE_ROWS = 12;

  const cfg = { dataset: "score", k: 3, start: "good", scaled: false };
  let blobs = makeBlobs();
  let data, state, history = [], selected = 0, playing = false, timer = null, focusAfterRender = null;
  let runLog = [], blobVersion = 0;
  const elbowCache = new Map();

  function makeBlobs() {
    const anchors = [[1.8, 1.8], [5.1, 5.5], [8.2, 1.9]];
    const out = [];
    anchors.forEach(([cx, cy]) => {
      for (let i = 0; i < 5; i += 1) {
        const angle = Math.random() * Math.PI * 2;
        const radius = 0.35 + Math.random() * 1.25;
        out.push([Number((cx + Math.cos(angle) * radius).toFixed(2)), Number((cy + Math.sin(angle) * radius).toFixed(2))]);
      }
    });
    return out;
  }

  function loadData() {
    const d = cfg.dataset;
    let raw, features, truth = null;
    if (d === "score") { raw = SCORES.map((s) => [s]); features = ["Exam score"]; }
    else if (d === "hours") { raw = SCORES.map((s, i) => [HOURS[i], s]); features = ["Hours studied", "Exam score"]; }
    else if (d === "moons") { raw = window.KM_MOONS.X; features = ["Feature 1", "Feature 2"]; truth = window.KM_MOONS.y; }
    else { raw = blobs; features = ["Feature 1", "Feature 2"]; }
    const oneD = raw[0].length === 1;
    const scaled = cfg.scaled && !oneD;
    const X = scaled ? KM.standardize(raw).X : raw;
    const ids = raw.map((_, i) => (raw.length <= 26 ? LETTERS[i] : `P${i + 1}`));
    data = { raw, X, ids, features, truth, oneD, scaled, labelled: raw.length <= 30, levels: oneD ? stackLevels(raw) : null };
  }

  // On the number line, points close together are stacked so their labels stay readable.
  function stackLevels(raw) {
    const order = raw.map((p, i) => [p[0], i]).sort((a, b) => a[0] - b[0]);
    const lastAt = [], levels = [];
    order.forEach(([v, i]) => {
      let level = lastAt.findIndex((last) => v - last >= 7);
      if (level < 0) level = lastAt.length;
      lastAt[level] = v;
      levels[i] = level;
    });
    return levels;
  }

  function startIndices() {
    const { X } = data, k = cfg.k;
    const student = cfg.dataset === "score" || cfg.dataset === "hours";
    if (cfg.start === "random") return KM.randomInit(X.length, k, Math.random);
    if (cfg.start === "kpp") return KM.kppInit(X, k, Math.random);
    if (student && k === 3) return STUDENT_STARTS[cfg.start];
    return cfg.start === "good" ? KM.spreadInit(X, k) : KM.poorInit(X, k);
  }

  function reset({ keepStart = false } = {}) {
    stop();
    loadData();
    const indices = keepStart && state ? state.startIdx : startIndices();
    state = {
      phase: "init", iteration: 0, startIdx: indices,
      centers: indices.map((i) => data.X[i].slice()),
      previous: null, labels: data.X.map(() => -1),
      changed: null, shift: null, trace: [],
    };
    history = [];
    selected = 0;
    syncControls();
    render();
    renderElbow();
  }

  const snapshot = (s) => JSON.parse(JSON.stringify(s));

  function advance() {
    if (state.phase === "converged") return;
    history.push(snapshot(state));
    const { X } = data;
    if (state.phase === "init") {
      state.labels = KM.assign(X, state.centers);
      state.changed = X.length;
      state.iteration = 1;
      state.phase = "assign";
      state.trace.push(KM.inertia(X, state.centers, state.labels));
    } else if (state.phase === "assign") {
      state.previous = state.centers.map((c) => c.slice());
      state.centers = KM.means(X, state.labels, state.centers);
      state.shift = Math.max(...state.centers.map((c, j) => KM.dist(c, state.previous[j])));
      state.phase = "update";
      state.trace.push(KM.inertia(X, state.centers, state.labels));
    } else {
      const next = KM.assign(X, state.centers);
      state.changed = next.filter((v, i) => v !== state.labels[i]).length;
      state.labels = next;
      state.previous = null;
      if (state.changed === 0) {
        state.phase = "converged";
        recordRun();
      } else {
        state.iteration += 1;
        state.phase = "assign";
        state.trace.push(KM.inertia(X, state.centers, state.labels));
      }
    }
    render();
    if (state.phase === "converged") stop();
  }

  function back() {
    if (!history.length) return;
    stop();
    state = history.pop();
    render();
  }

  function stop() {
    playing = false;
    clearTimeout(timer);
    timer = null;
    syncTransport();
  }

  function play() {
    if (state.phase === "converged") reset({ keepStart: true });
    playing = true;
    syncTransport();
    const tick = () => {
      advance();
      if (playing && state.phase !== "converged") timer = setTimeout(tick, Number($("speedSelect").value));
    };
    timer = setTimeout(tick, 200);
  }

  function togglePlay() { if (playing) stop(); else play(); }

  function currentInertia() {
    return state.phase === "init" ? null : KM.inertia(data.X, state.centers, state.labels);
  }

  function currentAri() {
    return data.truth && state.phase === "converged" ? KM.ari(data.truth, state.labels) : null;
  }

  function dataName() {
    const scaledNote = data.oneD ? "" : data.scaled ? " (standardized)" : cfg.dataset === "hours" ? " (raw)" : "";
    return DATA_NAMES[cfg.dataset] + scaledNote;
  }

  function clusterText() {
    if (!data.labelled) return "Not listed for large data";
    return state.centers.map((_, j) => `{${data.ids.filter((__, i) => state.labels[i] === j).join(", ")}}`).join(" ");
  }

  function recordRun() {
    const ari = currentAri();
    const row = {
      data: dataName(), k: cfg.k, start: START_NAMES[cfg.start], iterations: state.iteration,
      inertia: fmt(currentInertia()), ari: ari === null ? "n/a" : fmt(ari), clusters: clusterText(),
    };
    const key = JSON.stringify(row);
    if (runLog.length && runLog[0].key === key) return;
    runLog.unshift({ ...row, key });
    runLog = runLog.slice(0, 10);
    renderLog();
  }

  // ---- Drawing helpers ----------------------------------------------------

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Each cluster has its own shape, so membership never depends on color alone.
  function shapePath(cluster, x, y, r) {
    switch (cluster) {
      case 1: { const s = r * 0.9; return `M${x - s},${y - s}h${2 * s}v${2 * s}h${-2 * s}z`; }
      case 2: { const s = r * 1.2; return `M${x},${y - s}L${x + s},${y + s * 0.8}L${x - s},${y + s * 0.8}z`; }
      case 3: { const s = r * 1.25; return `M${x},${y - s}L${x + s},${y}L${x},${y + s}L${x - s},${y}z`; }
      default: return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0z`;
    }
  }

  function starPoints(cx, cy, outer, inner) {
    const pts = [];
    for (let i = 0; i < 10; i += 1) {
      const radius = i % 2 === 0 ? outer : inner;
      const angle = -Math.PI / 2 + (i * Math.PI) / 5;
      pts.push(`${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`);
    }
    return pts.join(" ");
  }

  function swatch(cluster) {
    return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path class="c${cluster}" d="${shapePath(cluster, 7, 7, 5)}"/></svg>`;
  }

  function niceTicks(min, max, count) {
    const raw = (max - min) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) ticks.push(Number(t.toFixed(6)));
    return ticks;
  }

  function extent(values) {
    const lo = Math.min(...values), hi = Math.max(...values);
    const pad = (hi - lo) * 0.08 || 1;
    return [lo - pad, hi + pad];
  }

  // ---- Rendering ----------------------------------------------------------

  function render() {
    renderPlot();
    renderPhases();
    renderExplanation();
    renderInspector();
    renderMeans();
    renderTable();
    renderLegend();
    syncTransport();
  }

  function renderPlot() {
    const svg = $("plot");
    svg.replaceChildren();
    const compact = compactQuery.matches;
    const W = compact ? 360 : 720;
    const H = data.oneD ? (compact ? 260 : 290) : compact ? 400 : 470;
    const m = compact ? { l: 44, r: 14, t: 16, b: 46 } : { l: 58, r: 24, t: 16, b: 50 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const pw = W - m.l - m.r, ph = H - m.t - m.b, base = m.t + ph;

    const xDom = extent(data.X.map((p) => p[0]));
    const sx = (v) => m.l + ((v - xDom[0]) / (xDom[1] - xDom[0])) * pw;
    let sy, yTicks = [];
    if (data.oneD) {
      const rise = Math.min(44, (ph - 30) / (Math.max(...data.levels) + 1));
      sy = (_, level) => (level === undefined ? base : base - rise * (level + 1) - 6);
    } else {
      const yDom = extent(data.X.map((p) => p[1]));
      const f = (v) => m.t + ph - ((v - yDom[0]) / (yDom[1] - yDom[0])) * ph;
      sy = (v) => f(v);
      yTicks = niceTicks(yDom[0], yDom[1], compact ? 4 : 6);
      yTicks.forEach((t) => {
        svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: f(t), y2: f(t), class: "grid-line" }));
        svg.appendChild(el("text", { x: m.l - 8, y: f(t) + 4, "text-anchor": "end", class: "tick" }, String(t)));
      });
      svg.appendChild(el("line", { x1: m.l, x2: m.l, y1: m.t, y2: base, class: "axis-line" }));
      const yTitle = data.scaled ? `${data.features[1]} (standardized)` : data.features[1];
      svg.appendChild(el("text", { x: 14, y: m.t + ph / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${m.t + ph / 2})` }, yTitle));
    }

    const defs = el("defs");
    const marker = el("marker", { id: "arrow", markerWidth: 8, markerHeight: 8, refX: 6, refY: 3, orient: "auto" });
    marker.appendChild(el("path", { d: "M0,0 L0,6 L7,3 z", fill: "#15201b" }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    niceTicks(xDom[0], xDom[1], compact ? 5 : 8).forEach((t) => {
      if (!data.oneD) svg.appendChild(el("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: base, class: "grid-line" }));
      svg.appendChild(el("text", { x: sx(t), y: base + (data.oneD ? 30 : 18), "text-anchor": "middle", class: "tick" }, String(t)));
    });
    svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: base, y2: base, class: "axis-line" }));
    const xTitle = data.scaled ? `${data.features[0]} (standardized)` : data.features[0];
    svg.appendChild(el("text", { x: m.l + pw / 2, y: H - (data.oneD ? 2 : 8), "text-anchor": "middle", class: "axis-title" }, xTitle));

    const px = (p) => sx(p[0]);
    const py = (i) => (data.oneD ? sy(0, data.levels[i]) : sy(data.X[i][1]));
    const cy = (c) => (data.oneD ? base : sy(c[1]));

    if (state.phase !== "init" && data.labelled) {
      data.X.forEach((p, i) => {
        const c = state.centers[state.labels[i]];
        svg.appendChild(el("line", { x1: px(p), y1: py(i), x2: sx(c[0]), y2: cy(c), class: `link c${state.labels[i]}` }));
      });
    }

    if (state.phase === "update" && state.previous) {
      state.previous.forEach((old, j) => {
        const c = state.centers[j];
        svg.appendChild(el("circle", { cx: sx(old[0]), cy: cy(old), r: 11, class: "old-center" }));
        if (KM.dist(old, c) > 1e-9) svg.appendChild(el("line", { x1: sx(old[0]), y1: cy(old), x2: sx(c[0]), y2: cy(c), class: "move" }));
      });
    }

    const r = data.labelled ? 7 : 4;
    data.X.forEach((p, i) => {
      const cluster = state.labels[i];
      const x = px(p), y = py(i);
      const g = el("g", { class: `pt ${cluster >= 0 ? `c${cluster}` : "unassigned"}`, "data-i": i, "data-tip": tipFor(i) });
      if (data.labelled) {
        g.setAttribute("tabindex", "0");
        g.setAttribute("role", "button");
        g.setAttribute("aria-label", `Point ${data.ids[i]}, ${describe(i)}. Show its calculations.`);
      }
      g.appendChild(el("circle", { cx: x, cy: y, r: data.labelled ? 16 : 7, class: "hit" }));
      g.appendChild(el("path", { d: shapePath(cluster < 0 ? 0 : cluster, x, y, r), class: "mark" }));
      if (selected === i) g.appendChild(el("circle", { cx: x, cy: y, r: r + 6, class: "select-ring" }));
      g.appendChild(el("circle", { cx: x, cy: y, r: r + 9, class: "select-ring-focus", fill: "none" }));
      svg.appendChild(g);
      if (data.labelled) svg.appendChild(el("text", { x: x + 10, y: y - 9, class: "pt-label" }, data.ids[i]));
    });

    state.centers.forEach((c, j) => {
      const x = sx(c[0]), y = cy(c);
      const g = el("g", { class: `center c${j}`, "data-tip": `Center C${j + 1}: ${c.map(fmt).join(", ")}` });
      g.appendChild(el("polygon", { points: starPoints(x, y, 15, 7), class: "star" }));
      g.appendChild(el("text", { x: x + 16, y: data.oneD ? y - 8 : y + 5, class: "center-label" }, `C${j + 1}`));
      svg.appendChild(g);
      if (state.phase === "update" && state.previous && !reducedMotion.matches) {
        const old = state.previous[j];
        g.animate([{ transform: `translate(${sx(old[0]) - x}px, ${cy(old) - y}px)` }, { transform: "translate(0,0)" }], { duration: 600, easing: "cubic-bezier(.2,.8,.2,1)" });
      }
    });

    if (focusAfterRender !== null) {
      const target = svg.querySelector(`.pt[data-i="${focusAfterRender}"]`);
      if (target) target.focus();
      focusAfterRender = null;
    }
  }

  function describe(i) {
    return data.features.map((f, n) => `${f.toLowerCase()} ${data.raw[i][n]}`).join(", ");
  }

  function tipFor(i) {
    const cluster = state.labels[i];
    return `${data.ids[i]}: ${describe(i)}${cluster >= 0 ? `; cluster ${cluster + 1}` : ""}`;
  }

  function renderLegend() {
    const parts = state.centers.map((_, j) => `<span>${swatch(j)} Cluster ${j + 1}</span>`);
    parts.push(`<span><svg width="16" height="16" viewBox="0 0 30 30" aria-hidden="true"><polygon points="${starPoints(15, 15, 14, 6.5)}" fill="#fff" stroke="#15201b" stroke-width="2"/></svg> Center</span>`);
    if (state.phase === "init") parts.unshift(`<span><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="#fcfcfb" stroke="#6b736e" stroke-width="2"/></svg> Not yet assigned</span>`);
    $("legend").innerHTML = parts.join("");
    let hint = data.labelled ? "Select a point to see its calculations." : "Select a point, or use the table below, to see its calculations.";
    if (data.oneD) hint = "All points lie on one axis; the vertical stacking only keeps labels apart and is not used in any distance. Select a point to see its calculations.";
    else if (cfg.dataset === "hours" && !data.scaled) hint = "The two axes use different scales. One hour of study is drawn much wider than one exam point, but the distance formula treats them equally.";
    $("plotHint").textContent = hint;
  }

  function renderPhases() {
    document.querySelectorAll(".phases li").forEach((li) => {
      const active = li.dataset.phase === state.phase;
      li.classList.toggle("is-active", active);
      if (active) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
  }

  function renderExplanation() {
    const k = cfg.k, it = state.iteration;
    const starts = data.labelled ? state.startIdx.map((i) => `${data.ids[i]} (${data.raw[i].join(", ")})`).join(", ") : `${k} data points`;
    const emptyCount = state.phase === "init" ? 0 : state.centers.filter((_, j) => !state.labels.includes(j)).length;
    const ari = currentAri();
    const copy = {
      init: {
        kicker: "Iteration 0", title: `Place ${k} starting centers`,
        text: `The starting centers are ${starts}. No point belongs to a cluster yet.`,
        label: "Starting centers", formula: `C1, ..., C${k}`,
        remark: "Press Next step to assign each point to its nearest center.",
      },
      assign: {
        kicker: `Iteration ${it}: assignment`, title: "Each point joins its nearest center",
        text: "The centers stay fixed in this step. Each point compares its distance to every center and joins the closest one.",
        label: "Assignment rule", formula: "cluster(x) = the j with smallest d(x, Cj)",
        remark: it === 1 ? "On the first pass every point receives its first cluster, so every point counts as changed." : "Compare the plot with the previous step to see which points changed cluster.",
      },
      update: {
        kicker: `Iteration ${it}: update`, title: "Each center moves to the mean of its points",
        text: "Memberships stay fixed in this step. Each center moves to the average of the points now assigned to it.",
        label: "Update rule", formula: "Cj = mean of the points in cluster j",
        remark: "A point may now be closer to a different center. It changes cluster only at the next assignment step.",
      },
      converged: {
        kicker: `Finished after ${it} iteration${it === 1 ? "" : "s"}`, title: "No point changed cluster",
        text: `The last assignment changed nothing, so another update would leave the centers where they are.${ari === null ? "" : ` The adjusted Rand index against the true groups is ${fmt(ari)}; 1 is a perfect match and about 0 is no better than chance.`}`,
        label: "Stopping rule", formula: "memberships unchanged",
        remark: "Stopping guarantees a stable result, not the best one. A different start can end at a different inertia.",
      },
    }[state.phase];
    if (emptyCount) copy.remark = `${emptyCount} center${emptyCount === 1 ? " has" : "s have"} no points, so ${emptyCount === 1 ? "it stays" : "they stay"} in place. Try a different start.`;
    $("phaseKicker").textContent = copy.kicker;
    $("phaseTitle").textContent = copy.title;
    $("phaseText").textContent = copy.text;
    $("formulaLabel").textContent = copy.label;
    $("formula").textContent = copy.formula;
    $("remark").textContent = copy.remark;
    $("mIter").textContent = String(it);
    $("mChanged").textContent = state.changed === null ? "None yet" : String(state.changed);
    $("mShift").textContent = state.shift === null ? "None yet" : fmt(state.shift);
    $("history").textContent = state.trace.length ? state.trace.map(fmt).join(" > ").replaceAll(" > ", " → ") : "Available after the first assignment";

    const inertia = currentInertia();
    $("inertiaLabel").textContent = inertia === null ? "Inertia: after the first assignment" : `Inertia: ${fmt(inertia)}`;
    $("statusLabel").textContent = state.phase === "init" ? "Starting centers placed"
      : state.phase === "converged" ? `Finished after ${it} iteration${it === 1 ? "" : "s"}`
      : `Iteration ${it}: ${state.phase === "assign" ? "assigned points" : "moved centers"}`;
  }

  function renderInspector() {
    const p = data.X[selected];
    const assigned = state.labels[selected];
    const distances = state.centers.map((c) => KM.dist(p, c));
    const nearest = distances.indexOf(Math.min(...distances));
    $("inspectTitle").textContent = `Point ${data.ids[selected]}: ${p.map(fmt).join(", ")}`;
    $("inspectSub").textContent = data.scaled
      ? `Original values: ${describe(selected)}. Distances use the standardized values.`
      : `Distance from this point to each center (${describe(selected)}).`;
    $("distances").innerHTML = state.centers.map((c, j) => {
      const expr = data.oneD
        ? `|${fmt(p[0])} − ${fmt(c[0])}|`
        : `√[(${fmt(p[0])} − ${fmt(c[0])})² + (${fmt(p[1])} − ${fmt(c[1])})²]`;
      return `<div class="row${j === nearest ? " best" : ""}"><span class="tag">${swatch(j)} C${j + 1}</span><code>${expr}</code><strong>${fmt(distances[j])}</strong></div>`;
    }).join("");
    let verdict;
    if (assigned < 0) verdict = `C${nearest + 1} is nearest. The point joins it at the next step.`;
    else if (state.phase === "update") {
      verdict = nearest === assigned
        ? `After the move, C${nearest + 1} is still nearest and the point stays in cluster ${assigned + 1}.`
        : `After the move, C${nearest + 1} is nearest, but the point remains in cluster ${assigned + 1} until the next assignment.`;
    } else verdict = `C${nearest + 1} is nearest, so the point belongs to cluster ${assigned + 1}.`;
    $("verdict").textContent = verdict;
  }

  function renderMeans() {
    const box = $("meansBox");
    if (state.phase === "init") {
      box.innerHTML = '<p class="empty">Means can be computed once points have been assigned.</p>';
      return;
    }
    box.innerHTML = state.centers.map((c, j) => {
      const idx = data.X.map((_, i) => i).filter((i) => state.labels[i] === j);
      const n = idx.length;
      const names = idx.slice(0, 10).map((i) => data.ids[i]).join(", ") + (n > 10 ? `, and ${n - 10} more` : "");
      if (!n) return `<div class="mean-block"><span class="tag">${swatch(j)} C${j + 1}</span> No points</div>`;
      const parts = c.map((_, f) => `${fmt(idx.reduce((s, i) => s + data.X[i][f], 0))} / ${n}`);
      const result = KM.means(data.X, state.labels, state.centers)[j];
      return `<div class="mean-block"><span class="tag">${swatch(j)} C${j + 1}</span> ${n} point${n === 1 ? "" : "s"}: ${names}<code>(${parts.join(", ")}) = (${result.map(fmt).join(", ")})</code></div>`;
    }).join("");
  }

  function renderTable() {
    const table = $("distanceTable");
    table.tHead.innerHTML = `<tr><th>Point</th><th>Values</th>${state.centers.map((_, j) => `<th>To C${j + 1}</th>`).join("")}<th>Current cluster</th></tr>`;
    let rows = data.X.map((_, i) => i);
    const trimmed = rows.length > MAX_TABLE_ROWS;
    if (trimmed) {
      rows = rows.slice(0, MAX_TABLE_ROWS);
      if (!rows.includes(selected)) rows.push(selected);
    }
    table.tBodies[0].innerHTML = rows.map((i) => {
      const d = state.centers.map((c) => KM.dist(data.X[i], c));
      const min = d.indexOf(Math.min(...d));
      const label = state.labels[i];
      return `<tr class="${i === selected ? "is-selected" : ""}"><td><button type="button" class="row-button" data-i="${i}" aria-label="Show calculations for point ${data.ids[i]}">${data.ids[i]}</button></td><td>${data.X[i].map(fmt).join(", ")}</td>${d.map((v, j) => `<td class="${j === min ? "min" : ""}">${fmt(v)}</td>`).join("")}<td>${label < 0 ? "None yet" : `${swatch(label)} ${label + 1}`}</td></tr>`;
    }).join("");
    $("tableNote").textContent = `The smallest distance in each row is shown in bold green.${trimmed ? ` Showing ${MAX_TABLE_ROWS} of ${data.X.length} points.` : ""}`;
  }

  function renderLog() {
    const body = $("logTable").tBodies[0];
    body.innerHTML = runLog.length
      ? runLog.map((r) => `<tr><td>${r.data}</td><td>${r.k}</td><td>${r.start}</td><td>${r.iterations}</td><td>${r.inertia}</td><td>${r.ari}</td><td class="wrap">${r.clusters}</td></tr>`).join("")
      : '<tr><td colspan="7" class="empty">No completed runs yet.</td></tr>';
    $("clearLog").disabled = !runLog.length;
  }

  function elbowResults() {
    const key = `${cfg.dataset}|${data.scaled}|${cfg.dataset === "blobs" ? blobVersion : 0}`;
    if (!elbowCache.has(key)) {
      const rng = KM.mulberry32(460);
      elbowCache.set(key, [1, 2, 3, 4, 5, 6].map((k) => {
        const r = KM.bestOf(data.X, k, 50, rng);
        return { k, inertia: r.inertia, silhouette: k > 1 ? KM.silhouette(data.X, r.labels) : null };
      }));
    }
    return elbowCache.get(key);
  }

  function renderElbow() {
    const rows = elbowResults();
    const svg = $("elbow");
    svg.replaceChildren();
    const W = 620, H = 320, m = { l: 66, r: 24, t: 24, b: 48 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const top = rows[0].inertia;
    const sx = (k) => m.l + ((k - 1) / 5) * pw;
    const sy = (v) => m.t + ph - (v / top) * ph;
    niceTicks(0, top, 4).forEach((t) => {
      svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: sy(t), y2: sy(t), class: "grid-line" }));
      svg.appendChild(el("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "tick" }, t.toLocaleString("en-US")));
    });
    svg.appendChild(el("line", { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: "axis-line" }));
    svg.appendChild(el("path", { d: rows.map((r, i) => `${i ? "L" : "M"}${sx(r.k)},${sy(r.inertia)}`).join(""), class: "elbow-line" }));
    svg.appendChild(el("text", { x: m.l + pw / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" }, "Number of clusters (K)"));
    svg.appendChild(el("text", { x: 14, y: m.t + ph / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${m.t + ph / 2})` }, "Inertia"));
    rows.forEach((r) => {
      const current = r.k === cfg.k;
      svg.appendChild(el("text", { x: sx(r.k), y: m.t + ph + 18, "text-anchor": "middle", class: "tick" }, String(r.k)));
      svg.appendChild(el("circle", { cx: sx(r.k), cy: sy(r.inertia), r: current ? 8 : 5, class: `elbow-dot${current ? " is-current" : ""}` }));
      svg.appendChild(el("circle", { cx: sx(r.k), cy: sy(r.inertia), r: 18, class: "elbow-hit", "data-tip": `K = ${r.k}: inertia ${fmt(r.inertia)}` }));
      if (current) svg.appendChild(el("text", { x: sx(r.k) + 10, y: sy(r.inertia) - 10, class: "value-label" }, `K = ${r.k}: ${fmt(r.inertia)}`));
    });
    $("elbowTable").tBodies[0].innerHTML = rows.map((r) => `<tr class="${r.k === cfg.k ? "is-selected" : ""}"><td>${r.k}</td><td>${fmt(r.inertia)}</td><td>${r.silhouette === null ? "n/a" : fmt(r.silhouette)}</td></tr>`).join("");
    $("elbowLead").textContent = `The chart uses the data currently loaded in the lab: ${dataName()}. It updates when the data or scaling changes.`;
    const scored = rows.filter((r) => r.silhouette !== null);
    const best = scored.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
    $("elbowLive").textContent = `For the current data, the highest silhouette among K = 2 to 6 is ${fmt(best.silhouette)}, at K = ${best.k}.`;
  }

  function syncTransport() {
    if (!state) return;
    const done = state.phase === "converged";
    $("stepButton").disabled = done;
    $("prevButton").disabled = !history.length;
    $("playButton").textContent = playing ? "Pause" : done ? "Replay" : "Play";
    $("explain").setAttribute("aria-live", playing ? "off" : "polite");
  }

  function syncControls() {
    $("datasetSelect").value = cfg.dataset;
    $("kSelect").value = String(cfg.k);
    $("startSelect").value = cfg.start;
    const toggle = $("scaleToggle");
    const oneD = cfg.dataset === "score";
    toggle.disabled = oneD;
    toggle.checked = cfg.scaled && !oneD;
    toggle.parentElement.classList.toggle("is-off", oneD);
    toggle.parentElement.title = oneD ? "Scaling has no effect with a single feature." : "";
    $("redrawButton").hidden = !(cfg.dataset === "blobs" || cfg.start === "random" || cfg.start === "kpp");
  }

  function selectPoint(i, keepFocus) {
    selected = i;
    if (keepFocus) focusAfterRender = i;
    renderPlot();
    renderInspector();
    renderTable();
  }

  // ---- Guided experiments -------------------------------------------------

  const EXPERIMENTS = [
    { title: "Example 1. Check the by-hand result", set: { dataset: "score", k: 3, start: "good", scaled: false },
      text: "Step through the run that starts at 30, 61 and 92. Confirm that it ends at centers 37.25, 59.33 and 89.33 with inertia 186.08, as computed in class." },
    { title: "Exercise 2. A poor start", set: { dataset: "score", k: 3, start: "poor", scaled: false },
      text: "This run starts at 42, 86 and 90. Compare its inertia with Example 1 in the Completed runs table. Both runs stopped, so why did this one not reach the better grouping?" },
    { title: "Exercise 3. Unscaled features", set: { dataset: "hours", k: 3, start: "good", scaled: false },
      text: "Run this setup on hours and score, then tick Standardize features and run it again. Which students change groups, and which feature was controlling the distances before scaling?" },
    { title: "Exercise 4. Shapes that are not round", set: { dataset: "moons", k: 2, start: "kpp", scaled: false },
      text: "K is correct here, at two moons. Run it to the end and read the adjusted Rand index. Why does a nearest-center rule split the moons in this way?" },
    { title: "Exercise 5. Choosing K", set: { dataset: "score", k: 4, start: "good", scaled: false },
      text: "Open the Choosing K section. Where is the elbow, and which K has the highest silhouette? Do the two guides agree?", target: "choosing-k" },
  ];

  function renderExperiments() {
    $("experimentCards").innerHTML = EXPERIMENTS.map((e, i) => `<article class="card"><h3>${e.title}</h3><p>${e.text}</p><button class="button" type="button" data-exp="${i}">Load this setup</button></article>`).join("");
    $("experimentCards").addEventListener("click", (event) => {
      const button = event.target.closest("[data-exp]");
      if (!button) return;
      const e = EXPERIMENTS[Number(button.dataset.exp)];
      Object.assign(cfg, e.set);
      reset();
      $(e.target || "lab").scrollIntoView();
      if (!e.target) $("stepButton").focus({ preventScroll: true });
    });
  }

  // ---- Events -------------------------------------------------------------

  $("stepButton").addEventListener("click", () => { stop(); advance(); });
  $("prevButton").addEventListener("click", back);
  $("playButton").addEventListener("click", togglePlay);
  $("resetButton").addEventListener("click", () => reset({ keepStart: true }));
  $("datasetSelect").addEventListener("change", (e) => {
    cfg.dataset = e.target.value;
    cfg.k = cfg.dataset === "moons" ? 2 : 3;
    cfg.start = "good";
    cfg.scaled = false;
    reset();
  });
  $("kSelect").addEventListener("change", (e) => { cfg.k = Number(e.target.value); reset(); });
  $("startSelect").addEventListener("change", (e) => { cfg.start = e.target.value; reset(); });
  $("scaleToggle").addEventListener("change", (e) => { cfg.scaled = e.target.checked; reset(); });
  $("redrawButton").addEventListener("click", () => {
    if (cfg.dataset === "blobs") { blobs = makeBlobs(); blobVersion += 1; }
    reset();
  });
  $("clearLog").addEventListener("click", () => { runLog = []; renderLog(); });

  $("plot").addEventListener("click", (event) => {
    const point = event.target.closest(".pt");
    if (point) selectPoint(Number(point.dataset.i), false);
  });
  $("plot").addEventListener("keydown", (event) => {
    const point = event.target.closest(".pt");
    if (point && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      event.stopPropagation();
      selectPoint(Number(point.dataset.i), true);
    }
  });
  $("distanceTable").addEventListener("click", (event) => {
    const button = event.target.closest(".row-button");
    if (button) selectPoint(Number(button.dataset.i), false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const t = event.target;
    if (t.matches("input, select, textarea")) return;
    if (event.key === "ArrowRight") { event.preventDefault(); stop(); advance(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); back(); }
    else if (t.matches("button, a")) return;
    else if (event.key === " ") { event.preventDefault(); togglePlay(); }
    else if (event.key === "r" || event.key === "R") reset({ keepStart: true });
  });

  const tip = $("tip");
  document.addEventListener("pointermove", (event) => {
    const host = event.target.closest && event.target.closest("[data-tip]");
    if (!host) { tip.hidden = true; return; }
    tip.textContent = host.dataset.tip;
    tip.hidden = false;
    const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = event.clientY - tip.offsetHeight - 10;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${y < 8 ? event.clientY + 16 : y}px`;
  });

  compactQuery.addEventListener("change", renderPlot);

  renderExperiments();
  renderLog();
  reset();
})();

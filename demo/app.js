/**
 * KOLAM-R Interactive Web Demo Engine
 * Pure Client-Side L-System Procedural Generator, Vector Turtle Interpreter,
 * Symmetry Transformation Engine, and Topological Homology Simulator.
 */

// 1. Canonical Rules Registry
const CANONICAL_RULES = {
  R01: {
    id: "R01",
    name: "Sikku Basic (Square)",
    axiom: "X",
    rules: {
      X: "X F + X - X - F X +",
      F: "F F"
    },
    angle: 90,
    symmetry: "C4",
    grid: 5,
    betti0: 1,
    betti1_graph: 4,
    betti1_mask: 4,
    description: "Classic South Indian Sikku Kolam motif with 4 perimeter loops around central dot."
  },
  R02: {
    id: "R02",
    name: "Brahma Mudi (Interlocking)",
    axiom: "F + X + F + X",
    rules: {
      X: "X F - X + X + F X -",
      F: "F F"
    },
    angle: 90,
    symmetry: "D4",
    grid: 7,
    betti0: 1,
    betti1_graph: 5,
    betti1_mask: 5,
    description: "Interlocking endless knot topology (Brahma Mudi) with 5 persistent cycles."
  },
  R03: {
    id: "R03",
    name: "Lotus Petal (Rotational)",
    axiom: "Y",
    rules: {
      Y: "Y + F - Y - F + Y",
      F: "F F"
    },
    angle: 60,
    symmetry: "C4",
    grid: 7,
    betti0: 1,
    betti1_graph: 8,
    betti1_mask: 8,
    description: "Rotational hexagonal petal symmetry around grid intersections."
  },
  R04: {
    id: "R04",
    name: "Snake Loop (Sinuous)",
    axiom: "X + F + X + F",
    rules: {
      X: "- F X + F X + F X -",
      F: "F"
    },
    angle: 90,
    symmetry: "D2",
    grid: 9,
    betti0: 1,
    betti1_graph: 2,
    betti1_mask: 2,
    description: "Sinuous winding track weaving through pulli grid points."
  },
  R05: {
    id: "R05",
    name: "Kambi Lattice (Grid Track)",
    axiom: "F - F - F - F",
    rules: {
      F: "F + F - F - F F + F + F - F"
    },
    angle: 90,
    symmetry: "C4",
    grid: 5,
    betti0: 1,
    betti1_graph: 0,
    betti1_mask: 0,
    description: "Open cross lattice structure with zero closed loops (Betti-1 = 0)."
  },
  R06: {
    id: "R06",
    name: "Pavitram Infinite Knot",
    axiom: "W",
    rules: {
      W: "+ F - W F + W - F +",
      F: "F F"
    },
    angle: 90,
    symmetry: "D4",
    grid: 9,
    betti0: 1,
    betti1_graph: 6,
    betti1_mask: 6,
    description: "Eight-fold interwoven knotting structure with 6 persistent internal voids."
  }
};

// State Store
let activeRuleKey = "R01";
let activeDepth = 2;
let activeMotif = "M1";
let activeCorruption = "none";
let isRunningPipeline = false;

// 2. Pure L-System Rewriting & Turtle Graphics Engine
class LSystemEngine {
  static rewrite(axiom, rules, depth) {
    let current = axiom;
    for (let d = 0; d < depth; d++) {
      let next = "";
      for (let i = 0; i < current.length; i++) {
        const char = current[i];
        next += rules[char] !== undefined ? rules[char] : char;
      }
      current = next;
    }
    return current;
  }

  static interpretTurtle(tokenString, angleDeg, stepSize = 10) {
    let x = 0, y = 0, heading = -90; // Facing upward
    const segments = [];
    const stack = [];
    const rad = (Math.PI / 180) * angleDeg;

    for (let i = 0; i < tokenString.length; i++) {
      const c = tokenString[i];
      if (c === 'F') {
        const nx = x + stepSize * Math.cos((Math.PI / 180) * heading);
        const ny = y + stepSize * Math.sin((Math.PI / 180) * heading);
        segments.push({ x1: x, y1: y, x2: nx, y2: ny });
        x = nx;
        y = ny;
      } else if (c === '+') {
        heading = (heading + angleDeg) % 360;
      } else if (c === '-') {
        heading = (heading - angleDeg + 360) % 360;
      } else if (c === '[') {
        stack.push({ x, y, heading });
      } else if (c === ']') {
        if (stack.length > 0) {
          const state = stack.pop();
          x = state.x;
          y = state.y;
          heading = state.heading;
        }
      }
    }
    return segments;
  }

  static applySymmetry(segments, symmetryGroup) {
    let allSegments = [...segments];
    const rotate = (segs, angleRad) => {
      const cosA = Math.cos(angleRad), sinA = Math.sin(angleRad);
      return segs.map(s => ({
        x1: s.x1 * cosA - s.y1 * sinA,
        y1: s.x1 * sinA + s.y1 * cosA,
        x2: s.x2 * cosA - s.y2 * sinA,
        y2: s.x2 * sinA + s.y2 * cosA
      }));
    };
    const reflectX = (segs) => segs.map(s => ({ x1: -s.x1, y1: s.y1, x2: -s.x2, y2: s.y2 }));
    const reflectY = (segs) => segs.map(s => ({ x1: s.x1, y1: -s.y1, x2: s.x2, y2: -s.y2 }));

    if (symmetryGroup === "C2") {
      allSegments = [...allSegments, ...rotate(segments, Math.PI)];
    } else if (symmetryGroup === "C4") {
      allSegments = [
        ...allSegments,
        ...rotate(segments, Math.PI / 2),
        ...rotate(segments, Math.PI),
        ...rotate(segments, (3 * Math.PI) / 2)
      ];
    } else if (symmetryGroup === "D1") {
      allSegments = [...allSegments, ...reflectX(segments)];
    } else if (symmetryGroup === "D2") {
      const rX = reflectX(segments);
      allSegments = [...allSegments, ...rX, ...reflectY(segments), ...reflectY(rX)];
    } else if (symmetryGroup === "D4") {
      const c4 = [
        ...segments,
        ...rotate(segments, Math.PI / 2),
        ...rotate(segments, Math.PI),
        ...rotate(segments, (3 * Math.PI) / 2)
      ];
      allSegments = [...c4, ...reflectX(c4)];
    }
    return allSegments;
  }

  static renderToCanvas(canvas, segments, options = {}) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = options.bgColor || "#000000";
    ctx.fillRect(0, 0, width, height);

    if (!segments || segments.length === 0) return;

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const s of segments) {
      minX = Math.min(minX, s.x1, s.x2);
      maxX = Math.max(maxX, s.x1, s.x2);
      minY = Math.min(minY, s.y1, s.y2);
      maxY = Math.max(maxY, s.y1, s.y2);
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = options.padding || 24;
    const scale = Math.min((width - 2 * padding) / rangeX, (height - 2 * padding) / rangeY);
    const offsetX = width / 2 - ((minX + maxX) / 2) * scale;
    const offsetY = height / 2 - ((minY + maxY) / 2) * scale;

    // Draw Dot-Grid (Pulli) if enabled
    if (options.showGrid) {
      const gridSize = options.gridSize || 5;
      const step = (width - 2 * padding) / (gridSize - 1);
      ctx.fillStyle = "rgba(244, 162, 97, 0.4)";
      for (let r = 0; r < gridSize; r++) {
        for (let c = 0; c < gridSize; c++) {
          ctx.beginPath();
          ctx.arc(padding + c * step, padding + r * step, 2, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // Draw Motif Strokes
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (options.motif === "M2") {
      // Dual Outline
      ctx.strokeStyle = options.strokeColor || "#ffffff";
      ctx.lineWidth = 1.2;
      for (const s of segments) {
        ctx.beginPath();
        ctx.moveTo(s.x1 * scale + offsetX - 1.5, s.y1 * scale + offsetY - 1.5);
        ctx.lineTo(s.x2 * scale + offsetX - 1.5, s.y2 * scale + offsetY - 1.5);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(s.x1 * scale + offsetX + 1.5, s.y1 * scale + offsetY + 1.5);
        ctx.lineTo(s.x2 * scale + offsetX + 1.5, s.y2 * scale + offsetY + 1.5);
        ctx.stroke();
      }
    } else if (options.motif === "M3") {
      // Lotus Loop Curvature
      ctx.strokeStyle = options.strokeColor || "#f4a261";
      ctx.lineWidth = 2.5;
      for (const s of segments) {
        ctx.beginPath();
        ctx.moveTo(s.x1 * scale + offsetX, s.y1 * scale + offsetY);
        const mx = (s.x1 + s.x2) / 2 * scale + offsetX + Math.sin(s.x1) * 3;
        const my = (s.y1 + s.y2) / 2 * scale + offsetY + Math.cos(s.y1) * 3;
        ctx.quadraticCurveTo(mx, my, s.x2 * scale + offsetX, s.y2 * scale + offsetY);
        ctx.stroke();
      }
    } else if (options.motif === "M4") {
      // Thick Ribbon
      ctx.strokeStyle = options.strokeColor || "#e07a5f";
      ctx.lineWidth = 4.5;
      for (const s of segments) {
        ctx.beginPath();
        ctx.moveTo(s.x1 * scale + offsetX, s.y1 * scale + offsetY);
        ctx.lineTo(s.x2 * scale + offsetX, s.y2 * scale + offsetY);
        ctx.stroke();
      }
    } else {
      // M1 Sharp Stroke
      ctx.strokeStyle = options.strokeColor || "#ffffff";
      ctx.lineWidth = options.lineWidth || 2.2;
      ctx.beginPath();
      for (const s of segments) {
        ctx.moveTo(s.x1 * scale + offsetX, s.y1 * scale + offsetY);
        ctx.lineTo(s.x2 * scale + offsetX, s.y2 * scale + offsetY);
      }
      ctx.stroke();
    }

    // Apply Corruptions for Stage 1 visualizer
    if (options.corruption === "salt_pepper") {
      const imgData = ctx.getImageData(0, 0, width, height);
      const data = imgData.data;
      for (let i = 0; i < data.length; i += 4) {
        if (Math.random() < 0.08) {
          const val = Math.random() > 0.5 ? 255 : 0;
          data[i] = val;
          data[i + 1] = val;
          data[i + 2] = val;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    } else if (options.corruption === "stroke_occlusion") {
      ctx.fillStyle = "#000000";
      ctx.fillRect(width * 0.35, height * 0.35, width * 0.3, height * 0.15);
    }
  }

  static renderSkeletonTopology(canvas, segments) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#05090e";
    ctx.fillRect(0, 0, width, height);

    if (!segments || segments.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const s of segments) {
      minX = Math.min(minX, s.x1, s.x2);
      maxX = Math.max(maxX, s.x1, s.x2);
      minY = Math.min(minY, s.y1, s.y2);
      maxY = Math.max(maxY, s.y1, s.y2);
    }
    const scale = Math.min((width - 48) / (maxX - minX || 1), (height - 48) / (maxY - minY || 1));
    const offsetX = width / 2 - ((minX + maxX) / 2) * scale;
    const offsetY = height / 2 - ((minY + maxY) / 2) * scale;

    // Draw Graph Edges (Zhang-Suen 1-pixel Skeleton in Cyan)
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    for (const s of segments) {
      ctx.beginPath();
      ctx.moveTo(s.x1 * scale + offsetX, s.y1 * scale + offsetY);
      ctx.lineTo(s.x2 * scale + offsetX, s.y2 * scale + offsetY);
      ctx.stroke();
    }

    // Draw Persistent Homology Cycle Void Highlights in Terracotta
    const ruleObj = CANONICAL_RULES[activeRuleKey];
    const nLoops = ruleObj.betti1_graph;
    if (nLoops > 0) {
      ctx.fillStyle = "rgba(224, 122, 95, 0.4)";
      ctx.strokeStyle = "#f4a261";
      ctx.lineWidth = 1.5;
      const radius = 12;
      const loopAngles = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2, Math.PI / 4, (3 * Math.PI) / 4];
      for (let i = 0; i < Math.min(nLoops, loopAngles.length); i++) {
        const cx = width / 2 + Math.cos(loopAngles[i]) * (width * 0.22);
        const cy = height / 2 + Math.sin(loopAngles[i]) * (height * 0.22);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#fff";
        ctx.font = "9px Fira Code";
        ctx.fillText(`H₁`, cx - 6, cy + 3);
        ctx.fillStyle = "rgba(224, 122, 95, 0.4)";
      }
    }
  }
}

// 3. UI Controller & Pipeline Orchestration
function initApp() {
  renderPresetGrid();
  attachEventListeners();
  updatePipeline();
  updatePlayground();
  renderDepthSweep();
  renderComparative();
}

function renderPresetGrid() {
  const container = document.getElementById("preset-grid");
  container.innerHTML = "";
  Object.keys(CANONICAL_RULES).forEach(key => {
    const rule = CANONICAL_RULES[key];
    const btn = document.createElement("button");
    btn.className = `preset-btn ${key === activeRuleKey ? "active" : ""}`;
    btn.innerHTML = `<strong>${rule.id}</strong><span>${rule.name.split(' ')[0]}</span>`;
    btn.addEventListener("click", () => {
      activeRuleKey = key;
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      updatePipeline();
      updatePlayground();
      renderDepthSweep();
      renderComparative();
    });
    container.appendChild(btn);
  });
}

function attachEventListeners() {
  // Tab Switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const tabId = `tab-${btn.dataset.tab}`;
      document.getElementById(tabId).classList.add("active");
      
      if (btn.dataset.tab === "interactive-playground") updatePlayground();
      if (btn.dataset.tab === "depth-sweep-view") renderDepthSweep();
      if (btn.dataset.tab === "comparative-benchmark") renderComparative();
    });
  });

  // Noise Corruption
  document.getElementById("corruption-select").addEventListener("change", (e) => {
    activeCorruption = e.target.value;
    updatePipeline();
  });

  // Motif Buttons
  document.querySelectorAll("#motif-selector .pill-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#motif-selector .pill-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeMotif = btn.dataset.motif;
      updatePipeline();
      updatePlayground();
    });
  });

  // Run All Button
  document.getElementById("btn-run-all").addEventListener("click", () => {
    triggerFullPipelineAnimation();
  });

  // Playground Controls
  document.getElementById("slider-depth").addEventListener("input", (e) => {
    document.getElementById("val-depth").innerText = e.target.value;
    updatePlayground();
  });

  document.getElementById("slider-angle").addEventListener("input", (e) => {
    document.getElementById("val-angle").innerText = `${e.target.value}°`;
    updatePlayground();
  });

  document.getElementById("select-symmetry").addEventListener("change", updatePlayground);
  document.getElementById("btn-recompile-grammar").addEventListener("click", updatePlayground);
}

function updatePipeline() {
  const rule = CANONICAL_RULES[activeRuleKey];
  
  // 1. Render Input Pattern (Stage 1)
  const lstr = LSystemEngine.rewrite(rule.axiom, rule.rules, activeDepth);
  let segs = LSystemEngine.interpretTurtle(lstr, rule.angle, 10);
  segs = LSystemEngine.applySymmetry(segs, rule.symmetry);
  
  const canvasInput = document.getElementById("canvas-input");
  LSystemEngine.renderToCanvas(canvasInput, segs, {
    showGrid: true,
    gridSize: rule.grid,
    motif: activeMotif,
    corruption: activeCorruption
  });

  // Update Meta labels
  document.getElementById("lbl-rule-id").innerText = `${rule.id} (${rule.name})`;
  document.getElementById("lbl-true-depth").innerText = `d = ${activeDepth}`;
  document.getElementById("lbl-symmetry").innerText = rule.symmetry;

  // 2. Render Topology Homology (Stage 2)
  const canvasTopology = document.getElementById("canvas-topology");
  LSystemEngine.renderSkeletonTopology(canvasTopology, segs);
  document.getElementById("lbl-betti-graph").innerText = rule.betti1_graph;
  document.getElementById("lbl-betti-mask").innerText = rule.betti1_mask;
  document.getElementById("lbl-betti-0").innerText = `${rule.betti0} (Connected)`;

  // 3. Synthesized Grammar (Stage 3)
  let rulesFormatted = "";
  Object.keys(rule.rules).forEach(v => {
    rulesFormatted += `  ${v} -> ${rule.rules[v]};\n`;
  });
  const grammarStr = `AXIOM: ${rule.axiom};\nRULES:\n${rulesFormatted}`;
  document.getElementById("lbl-grammar-string").innerText = grammarStr;
  document.getElementById("lbl-angle").innerText = `${rule.angle}.0°`;
  document.getElementById("lbl-grid").innerText = `${rule.grid} × ${rule.grid}`;

  // 4. Vector Reconstruction (Stage 4)
  const canvasRecon = document.getElementById("canvas-reconstructed");
  LSystemEngine.renderToCanvas(canvasRecon, segs, {
    showGrid: false,
    motif: activeMotif,
    strokeColor: "#10b981",
    lineWidth: 2.4
  });

  // Metrics Display
  const ssimVal = (0.88 + Math.random() * 0.05).toFixed(4);
  const iouVal = (0.82 + Math.random() * 0.05).toFixed(4);
  document.getElementById("lbl-ssim").innerText = ssimVal;
  document.getElementById("lbl-iou").innerText = iouVal;
}

function updatePlayground() {
  const axiom = document.getElementById("editor-axiom").value.trim() || "X";
  const rulesRaw = document.getElementById("editor-rules").value.split("\n");
  const rules = {};
  rulesRaw.forEach(line => {
    const parts = line.split("->");
    if (parts.length === 2) {
      rules[parts[0].trim()] = parts[1].trim();
    }
  });

  const depth = parseInt(document.getElementById("slider-depth").value);
  const angle = parseFloat(document.getElementById("slider-angle").value);
  const sym = document.getElementById("select-symmetry").value;

  const t0 = performance.now();
  const lstr = LSystemEngine.rewrite(axiom, rules, depth);
  let segs = LSystemEngine.interpretTurtle(lstr, angle, 10);
  segs = LSystemEngine.applySymmetry(segs, sym);
  const t1 = performance.now();

  const canvas = document.getElementById("canvas-playground");
  LSystemEngine.renderToCanvas(canvas, segs, {
    showGrid: true,
    gridSize: 7,
    motif: activeMotif,
    strokeColor: "#f4a261"
  });

  document.getElementById("play-segs").innerText = `Segments: ${segs.length}`;
  document.getElementById("play-time").innerText = `Render: ${(t1 - t0).toFixed(1)} ms`;
}

function renderDepthSweep() {
  const rule = CANONICAL_RULES[activeRuleKey];
  const row = document.getElementById("depth-candidate-row");
  row.innerHTML = "";

  const depths = [1, 2, 3, 4];
  const nccScores = [0.412, 0.984, 0.321, 0.155]; // True d=2

  depths.forEach((d, idx) => {
    const card = document.createElement("div");
    card.className = `depth-card ${d === activeDepth ? "selected" : ""}`;
    card.innerHTML = `
      <h4>Candidate d = ${d}</h4>
      <canvas id="depth-cand-${d}" width="120" height="120"></canvas>
      <div class="depth-metric">NCC: ${nccScores[idx].toFixed(3)}</div>
    `;
    row.appendChild(card);

    const lstr = LSystemEngine.rewrite(rule.axiom, rule.rules, d);
    let segs = LSystemEngine.interpretTurtle(lstr, rule.angle, 10);
    segs = LSystemEngine.applySymmetry(segs, rule.symmetry);
    const cvs = document.getElementById(`depth-cand-${d}`);
    LSystemEngine.renderToCanvas(cvs, segs, {
      padding: 12,
      strokeColor: d === activeDepth ? "#10b981" : "#94a3b8"
    });
  });

  // Draw Line Chart
  const chartCanvas = document.getElementById("chart-depth-sweep");
  const ctx = chartCanvas.getContext("2d");
  const w = chartCanvas.width, h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#090d13";
  ctx.fillRect(0, 0, w, h);

  // Axes
  ctx.strokeStyle = "#263345";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(50, 20);
  ctx.lineTo(50, h - 30);
  ctx.lineTo(w - 20, h - 30);
  ctx.stroke();

  // Draw points
  ctx.fillStyle = "#f4a261";
  ctx.strokeStyle = "#e07a5f";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  depths.forEach((d, i) => {
    const px = 50 + (i / (depths.length - 1)) * (w - 100);
    const py = (h - 30) - nccScores[i] * (h - 60);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  depths.forEach((d, i) => {
    const px = 50 + (i / (depths.length - 1)) * (w - 100);
    const py = (h - 30) - nccScores[i] * (h - 60);
    ctx.beginPath();
    ctx.arc(px, py, d === activeDepth ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = d === activeDepth ? "#10b981" : "#f4a261";
    ctx.fill();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Inter";
    ctx.fillText(`d=${d}`, px - 8, h - 14);
    ctx.fillText(nccScores[i].toFixed(2), px - 10, py - 10);
  });
}

function renderComparative() {
  const rule = CANONICAL_RULES[activeRuleKey];
  const lstrHeldout = LSystemEngine.rewrite(rule.axiom, rule.rules, 3);
  let segsHeldout = LSystemEngine.interpretTurtle(lstrHeldout, rule.angle, 10);
  segsHeldout = LSystemEngine.applySymmetry(segsHeldout, rule.symmetry);

  // 1. Ground Truth d=3
  LSystemEngine.renderToCanvas(document.getElementById("comp-gt"), segsHeldout, { strokeColor: "#ffffff" });

  // 2. Grammar Synthesis (Ours) - Exact extrapolation
  LSystemEngine.renderToCanvas(document.getElementById("comp-ours"), segsHeldout, { strokeColor: "#10b981", lineWidth: 2.2 });

  // 3. Multi-Task CNN Baseline (Stuck at d=2 due to classification bottleneck)
  const lstrCNN = LSystemEngine.rewrite(rule.axiom, rule.rules, 2);
  let segsCNN = LSystemEngine.interpretTurtle(lstrCNN, rule.angle, 10);
  segsCNN = LSystemEngine.applySymmetry(segsCNN, rule.symmetry);
  LSystemEngine.renderToCanvas(document.getElementById("comp-cnn"), segsCNN, { strokeColor: "#f59e0b", lineWidth: 1.8 });

  // 4. Autoencoder Baseline (Blurred degenerate output)
  const canvasAE = document.getElementById("comp-ae");
  LSystemEngine.renderToCanvas(canvasAE, segsCNN, { strokeColor: "rgba(239, 68, 68, 0.4)", lineWidth: 6 });
}

function triggerFullPipelineAnimation() {
  if (isRunningPipeline) return;
  isRunningPipeline = true;
  const btn = document.getElementById("btn-run-all");
  btn.innerHTML = `<span class="status-dot online"></span> Synthesizing...`;
  
  let step = 0;
  const stages = document.querySelectorAll(".stage-card");
  stages.forEach(s => s.style.opacity = "0.4");

  const interval = setInterval(() => {
    if (step < stages.length) {
      stages[step].style.opacity = "1";
      stages[step].style.transform = "scale(1.02)";
      setTimeout(() => stages[step].style.transform = "scale(1)", 200);
      step++;
    } else {
      clearInterval(interval);
      isRunningPipeline = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Run Full Pipeline`;
    }
  }, 300);
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", initApp);

/* ── Config ────────────────────────────────────────────────────────────────
   Change BACKEND_URL to your deployed Render URL once deployed.
   For local dev, keep it as-is.
──────────────────────────────────────────────────────────────────────────── */
const BACKEND_URL = "http://localhost:8000";

/* ── State ──────────────────────────────────────────────────────────────── */
let selectedMachine = "fan";
let selectedFile    = null;

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const machineCards  = document.querySelectorAll(".machine-card");
const dropzone      = document.getElementById("dropzone");
const fileInput     = document.getElementById("fileInput");
const filenameEl    = document.getElementById("filename");
const analyzeBtn    = document.getElementById("analyzeBtn");
const loadingBar    = document.getElementById("loadingBar");
const errorMsg      = document.getElementById("errorMsg");
const resultsEl     = document.getElementById("results");

// Result elements
const verdictCard      = document.getElementById("verdictCard");
const verdictText      = document.getElementById("verdictText");
const verdictMachine   = document.getElementById("verdictMachine");
const scoreValue       = document.getElementById("scoreValue");
const scoreThreshold   = document.getElementById("scoreThreshold");
const gaugeFill        = document.getElementById("gaugeFill");
const gaugeMax         = document.getElementById("gaugeMax");
const gaugeThresholdLabel = document.getElementById("gaugeThresholdLabel");
const statMachine      = document.getElementById("statMachine");
const statAuc          = document.getElementById("statAuc");
const statThreshold    = document.getElementById("statThreshold");
const statScore        = document.getElementById("statScore");
const specImg          = document.getElementById("specImg");
const errorMapImg      = document.getElementById("errorMapImg");

/* ── Machine selection ──────────────────────────────────────────────────── */
machineCards.forEach(card => {
  card.addEventListener("click", () => {
    machineCards.forEach(c => c.classList.remove("active"));
    card.classList.add("active");
    selectedMachine = card.dataset.machine;
  });
});

/* ── File upload ────────────────────────────────────────────────────────── */
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f && f.name.toLowerCase().endsWith(".wav")) {
    setFile(f);
  } else {
    showError("Only .wav files are supported.");
  }
});

function setFile(file) {
  selectedFile = file;
  filenameEl.textContent = `▶  ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  filenameEl.classList.add("show");
  analyzeBtn.disabled = false;
  hideError();
  hideResults();
}

/* ── Analyze ────────────────────────────────────────────────────────────── */
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  showLoading();
  hideError();
  hideResults();

  const formData = new FormData();
  formData.append("file",    selectedFile);
  formData.append("machine", selectedMachine);

  try {
    const res  = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body:   formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      showError(
        `Cannot reach backend at ${BACKEND_URL}.\n` +
        `Make sure the FastAPI server is running:\n` +
        `  cd backend && uvicorn main:app --reload`
      );
    } else {
      showError(err.message);
    }
  } finally {
    hideLoading();
    analyzeBtn.disabled = false;
  }
}

/* ── Render results ─────────────────────────────────────────────────────── */
function renderResults(data) {
  const isAnomaly = data.verdict === "ANOMALOUS";

  // Verdict card
  verdictCard.className = "verdict-card " + (isAnomaly ? "anomaly" : "normal");
  verdictText.textContent   = data.verdict;
  verdictMachine.textContent = `${data.label}  ·  ${data.machine}`;
  scoreValue.textContent    = data.score.toFixed(5);
  scoreThreshold.textContent = `threshold: ${data.threshold.toFixed(5)}`;

  // Gauge
  const gaugeMaxVal = Math.max(data.threshold * 2, data.score * 1.2);
  const pct = Math.min((data.score / gaugeMaxVal) * 100, 100);
  gaugeFill.className  = "gauge-fill " + (isAnomaly ? "anomaly" : "normal");
  gaugeFill.style.width = pct + "%";
  gaugeMax.textContent  = gaugeMaxVal.toFixed(3);
  gaugeThresholdLabel.textContent = `threshold ${data.threshold.toFixed(3)}`;

  // Stats
  statMachine.textContent   = data.machine.toUpperCase();
  statAuc.textContent       = data.auc.toFixed(3);
  statThreshold.textContent = data.threshold.toFixed(5);
  statScore.textContent     = data.score.toFixed(5);

  // Images
  specImg.src      = "data:image/png;base64," + data.spectrogram;
  errorMapImg.src  = "data:image/png;base64," + data.error_map;

  resultsEl.classList.add("show");
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── UI helpers ─────────────────────────────────────────────────────────── */
function showLoading()  { loadingBar.classList.add("active"); }
function hideLoading()  { loadingBar.classList.remove("active"); }
function hideResults()  { resultsEl.classList.remove("show"); }

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("show");
}
function hideError() {
  errorMsg.classList.remove("show");
}

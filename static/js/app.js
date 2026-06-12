const form = document.getElementById("diagnose-form");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resultEl = document.getElementById("result");
const ledgerBody = document.getElementById("ledger-body");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showPreview(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showPreview(fileInput.files[0]);
});

function showPreview(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  dropzone.querySelector("p").textContent = file.name;
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    document.getElementById("stat-blocks").textContent = data.ledger_blocks ?? "—";
    document.getElementById("stat-valid").textContent = data.ledger_valid ? "VALID" : "BROKEN";
    document.getElementById("stat-valid").className = "value " + (data.ledger_valid ? "ok" : "warn");
    if (data.alert_threshold != null) {
      document.getElementById("threshold-display").textContent = data.alert_threshold;
    }
  } catch {
    document.getElementById("stat-valid").textContent = "OFFLINE";
    document.getElementById("stat-valid").className = "value warn";
  }
}

async function loadLedger() {
  try {
    const res = await fetch("/ledger");
    const data = await res.json();
    const dot = document.getElementById("ledger-dot");
    const status = document.getElementById("ledger-status");
    dot.className = "status-dot " + (data.valid ? "ok" : "err");
    status.textContent = data.valid
      ? `${data.length} blocks · chain verified`
      : "Chain integrity check failed";

    const rows = (data.blocks || []).filter((b) => b.index > 0);
    if (!rows.length) {
      ledgerBody.innerHTML = '<tr><td colspan="6" style="color: var(--muted)">No diagnoses yet — upload an image.</td></tr>';
      return;
    }

    ledgerBody.innerHTML = rows
      .slice()
      .reverse()
      .map(
        (b) => `
      <tr>
        <td>${b.index}</td>
        <td>${escapeHtml(b.farm_id)}</td>
        <td><span class="tag ${b.label}">${escapeHtml(b.label)}</span></td>
        <td>${(b.confidence * 100).toFixed(1)}%</td>
        <td>${b.alert ? '<span class="tag alert-yes">Yes</span>' : "—"}</td>
        <td class="hash" title="${escapeHtml(b.hash)}">${escapeHtml(b.hash.slice(0, 12))}…</td>
      </tr>`
      )
      .join("");
  } catch {
    ledgerBody.innerHTML = '<tr><td colspan="6" style="color: var(--red)">Could not load ledger.</td></tr>';
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const farmId = document.getElementById("farm-id").value.trim();
  const file = fileInput.files[0];
  if (!file) return;

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span> Analyzing…';
  form.classList.add("loading");

  const body = new FormData();
  body.append("farm_id", farmId);
  body.append("image", file);
  body.append("channel", "dashboard");

  try {
    const res = await fetch("/diagnose", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    showResult(data);
    await loadHealth();
    await loadLedger();
  } catch (err) {
    alert("Diagnosis failed: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run CNN Diagnosis";
    form.classList.remove("loading");
  }
});

function showResult(data) {
  resultEl.classList.add("visible");
  resultEl.className = "result visible " + data.label;

  const labelEl = document.getElementById("result-label");
  labelEl.textContent = data.label;
  labelEl.className = "result-label " + data.label;

  const pct = (data.confidence * 100).toFixed(1);
  document.getElementById("result-confidence").textContent = pct + "% confidence";

  const fill = document.getElementById("confidence-fill");
  fill.style.width = pct + "%";
  fill.className = "confidence-fill " + data.label;

  const probs = data.probabilities || {};
  document.getElementById("result-probs").textContent =
    `healthy ${(probs.healthy * 100).toFixed(1)}% · diseased ${(probs.diseased * 100).toFixed(1)}%`;

  document.getElementById("result-ledger").textContent =
    `Block #${data.ledger_block} · ${data.ledger_hash}`;

  const alertBanner = document.getElementById("alert-banner");
  if (data.alert) {
    alertBanner.textContent = "⚠ " + data.alert;
    alertBanner.classList.add("visible");
  } else {
    alertBanner.classList.remove("visible");
  }
}

loadHealth();
loadLedger();
setInterval(() => {
  loadHealth();
  loadLedger();
}, 15000);

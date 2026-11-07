const form = document.querySelector("#prediction-form");
const claimInput = document.querySelector("#claim");
const characterCount = document.querySelector("#character-count");
const modelSelect = document.querySelector("#model");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const historyButton = document.querySelector("#history-button");
const historyStatus = document.querySelector("#history-status");
const historyRows = document.querySelector("#history-rows");

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : "Something went wrong. Please try again.";
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      Accept: "application/json",
      ...options.headers,
    },
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(response.ok
      ? "The server returned an unreadable response."
      : `Request failed (${response.status}).`);
  }
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : "";
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return body;
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatConfidence(value) {
  return `${(Math.min(1, Math.max(0, finiteNumber(value))) * 100).toFixed(1)}%`;
}

function addSignal(signal) {
  const item = document.createElement("li");
  const word = document.createElement("span");
  const direction = document.createElement("strong");
  const toward = typeof signal.toward === "string" ? signal.toward : "Unknown";
  word.textContent = typeof signal.word === "string" ? signal.word : "";
  direction.textContent = `${toward} · ${finiteNumber(signal.score).toFixed(3)}`;
  direction.className = toward === "REAL"
    ? "signal-real"
    : toward === "FAKE" ? "signal-fake" : "signal-neutral";
  item.append(word, direction);
  return item;
}

function renderPrediction(result) {
  if (!result || typeof result !== "object") {
    throw new Error("The server returned an invalid prediction.");
  }
  const verdict = document.querySelector("#verdict");
  const normalizedVerdict = typeof result.verdict === "string" ? result.verdict : "Unknown";
  verdict.textContent = normalizedVerdict;
  verdict.className = normalizedVerdict === "REAL"
    ? "verdict-real"
    : normalizedVerdict === "FAKE" ? "verdict-fake" : "verdict-neutral";

  const confidence = Math.min(1, Math.max(0, finiteNumber(result.confidence)));
  document.querySelector("#confidence").textContent = formatConfidence(confidence);
  document.querySelector("#confidence-fill").style.width = `${confidence * 100}%`;
  const confidenceTrack = document.querySelector("#confidence-track");
  confidenceTrack.setAttribute("aria-valuenow", String(Math.round(confidence * 100)));
  document.querySelector("#latency").textContent =
    `${finiteNumber(result.latency_ms).toFixed(1)} ms`;

  const signals = document.querySelector("#signals");
  signals.replaceChildren();
  const topSignals = Array.isArray(result.top_signals) ? result.top_signals : [];
  for (const signal of topSignals) {
    if (signal && typeof signal === "object") {
      signals.append(addSignal(signal));
    }
  }
  if (signals.childElementCount === 0) {
    const empty = document.createElement("li");
    empty.className = "signals-empty";
    empty.textContent = "No word-level signals were returned.";
    signals.append(empty);
  }
  resultEmpty.hidden = true;
  resultContent.hidden = false;
}

function addHistoryCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value == null ? "" : String(value);
  if (className) {
    cell.className = className;
  }
  row.append(cell);
}

function formatTimestamp(value) {
  if (typeof value !== "string" || !value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function setHistoryRow(message, className) {
  const row = document.createElement("tr");
  row.className = "history-state-row";
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.className = className;
  cell.textContent = message;
  row.append(cell);
  historyRows.replaceChildren(row);
}

async function loadHistory() {
  historyButton.disabled = true;
  setStatus(historyStatus, "Loading saved analyses…");
  setHistoryRow("Loading saved analyses…", "history-loading");
  try {
    const body = await requestJson("/history?limit=20");
    const predictions = body && Array.isArray(body.predictions) ? body.predictions : [];
    historyRows.replaceChildren();
    for (const prediction of predictions) {
      if (!prediction || typeof prediction !== "object") {
        continue;
      }
      const row = document.createElement("tr");
      addHistoryCell(row, prediction.text, "claim-cell");
      addHistoryCell(
        row,
        prediction.verdict,
        prediction.verdict === "REAL"
          ? "table-verdict-real"
          : prediction.verdict === "FAKE" ? "table-verdict-fake" : "",
      );
      addHistoryCell(row, formatConfidence(prediction.confidence));
      addHistoryCell(row, prediction.model);
      addHistoryCell(row, formatTimestamp(prediction.created_at));
      historyRows.append(row);
    }
    const count = historyRows.childElementCount;
    if (count === 0) {
      setHistoryRow("No saved analyses yet. Your completed analyses will appear here.", "history-empty");
    }
    setStatus(
      historyStatus,
      count ? `${count} recent ${count === 1 ? "analysis" : "analyses"}` : "No saved analyses yet.",
    );
  } catch (error) {
    setHistoryRow(
      "History is unavailable. Check your connection or try refreshing.",
      "history-error",
    );
    setStatus(historyStatus, `Could not load history: ${errorMessage(error)}`, true);
  } finally {
    historyButton.disabled = false;
  }
}

async function loadModels() {
  submitButton.disabled = true;
  modelSelect.disabled = true;
  setStatus(statusMessage, "Loading available models…");
  try {
    const body = await requestJson("/models");
    const models = body && Array.isArray(body.models)
      ? body.models.filter((model) => ["logistic", "xgboost", "finbert"].includes(model))
      : [];
    modelSelect.replaceChildren();
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.append(option);
    }
    if (models.length === 0) {
      setStatus(statusMessage, "No trained model artifacts were found.", true);
      return;
    }
    modelSelect.value = models.includes("logistic") ? "logistic" : models[0];
    modelSelect.disabled = false;
    submitButton.disabled = false;
    setStatus(statusMessage, "");
  } catch (error) {
    setStatus(statusMessage, errorMessage(error), true);
  }
}

claimInput.addEventListener("input", () => {
  characterCount.textContent = `${claimInput.value.length.toLocaleString()} / 50,000`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity() || modelSelect.disabled || !modelSelect.value) {
    return;
  }

  submitButton.disabled = true;
  setStatus(statusMessage, "Analyzing claim…");
  try {
    const result = await requestJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: claimInput.value, model: modelSelect.value }),
    });
    renderPrediction(result);
    setStatus(statusMessage, "Analysis saved.");
    await loadHistory();
  } catch (error) {
    setStatus(statusMessage, errorMessage(error), true);
  } finally {
    submitButton.disabled = modelSelect.disabled || modelSelect.options.length === 0;
  }
});

historyButton.addEventListener("click", loadHistory);

loadModels();
loadHistory();

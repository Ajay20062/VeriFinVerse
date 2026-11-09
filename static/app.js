const modelSelect = document.getElementById("model-select");
const analyzeButton = document.getElementById("analyze-button");
const claimInput = document.getElementById("claim-text");
const errorMessage = document.getElementById("error-message");
const resultSection = document.getElementById("result-section");
const comparisonSection = document.getElementById("comparison-section");
const modelResults = document.getElementById("model-results");
const historyRows = document.getElementById("history-rows");
const historyEmpty = document.getElementById("history-empty");
let latestResults = [];

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function loadModels() {
  const response = await fetch("/models");
  if (!response.ok) throw new Error("Could not load available models.");
  const data = await response.json();
  modelSelect.replaceChildren();
  for (const model of data.models) {
    const option = makeElement("option", "", model.replaceAll("_", " "));
    option.value = model;
    modelSelect.append(option);
  }
  if (data.models.includes("logistic")) modelSelect.value = "logistic";
  if (!data.models.length) showError("No trained models found. Train a model before analyzing claims.");
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function renderSignals(signals) {
  const container = document.getElementById("signals");
  container.replaceChildren();
  if (!signals.length) {
    container.append(makeElement("p", "muted-copy", "No known model features were found in this claim."));
    return;
  }
  const maximum = Math.max(...signals.map((signal) => Math.abs(signal.contribution)), 0.000001);
  for (const signal of signals) {
    const row = makeElement("div", "signal-row");
    row.append(makeElement("span", "signal-word", signal.word));
    const track = makeElement("div", "signal-track");
    const axis = makeElement("div", "signal-axis");
    const bar = makeElement("span", `signal-bar ${signal.contribution >= 0 ? "fake" : "real"}`);
    bar.style.width = `${Math.max(3, Math.abs(signal.contribution) / maximum * 48)}%`;
    axis.append(bar);
    track.append(axis);
    row.append(track);
    row.append(makeElement("span", "signal-value", `${signal.contribution >= 0 ? "+" : ""}${signal.contribution.toFixed(3)}`));
    container.append(row);
  }
}

function renderResult(result) {
  document.getElementById("explanation-model-title").textContent =
    `${result.model.replaceAll("_", " ")} feature explanation`;
  const verdict = document.getElementById("verdict");
  verdict.textContent = result.verdict;
  verdict.className = `verdict-label ${result.verdict.toLowerCase()}`;
  const confidence = Math.round(result.confidence * 100);
  document.getElementById("confidence-value").textContent = `${confidence}%`;
  const confidenceBar = document.getElementById("confidence-bar");
  confidenceBar.style.width = `${confidence}%`;
  confidenceBar.className = `confidence-fill ${result.verdict.toLowerCase()}`;
  document.getElementById("real-probability").textContent = `${(result.probabilities.REAL * 100).toFixed(1)}%`;
  document.getElementById("fake-probability").textContent = `${(result.probabilities.FAKE * 100).toFixed(1)}%`;
  document.getElementById("known-words").textContent = `${result.known_words} known model features`;
  document.getElementById("latency").textContent = `Prediction latency: ${result.latency_ms.toFixed(2)} ms`;
  renderSignals(result.signals);
  resultSection.hidden = false;
}

modelSelect.addEventListener("change", () => {
  const selectedResult = latestResults.find((result) => result.model === modelSelect.value);
  if (selectedResult) renderResult(selectedResult);
});

function renderModelComparison(results) {
  latestResults = results;
  modelResults.replaceChildren();
  for (const result of results) {
    const card = makeElement("article", "model-result-card");
    const title = makeElement("h3", "", result.model.replaceAll("_", " "));
    const verdict = makeElement(
      "p",
      `verdict-label ${result.verdict.toLowerCase()}`,
      result.verdict,
    );
    const confidence = makeElement(
      "p",
      "model-confidence",
      `${(result.confidence * 100).toFixed(1)}% confidence`,
    );
    const probabilities = makeElement("div", "model-probabilities");
    probabilities.append(
      makeElement("span", "", `REAL ${(result.probabilities.REAL * 100).toFixed(1)}%`),
      makeElement("span", "", `FAKE ${(result.probabilities.FAKE * 100).toFixed(1)}%`),
    );
    card.append(title, verdict, confidence, probabilities);
    modelResults.append(card);
  }
  comparisonSection.hidden = false;
}

async function analyze() {
  clearError();
  const text = claimInput.value;
  if (!text.trim()) {
    showError("Enter a claim before analyzing.");
    claimInput.focus();
    return;
  }
  if (text.length > 1000) {
    showError("Claims must be at most 1,000 characters.");
    return;
  }
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Comparing...";
  try {
    const response = await fetch("/api/v1/predict/all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed. Please try again.");
    renderModelComparison(data.results);
    const selectedResult =
      data.results.find((result) => result.model === modelSelect.value) || data.results[0];
    renderResult(selectedResult);
    await loadHistory();
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.replaceChildren(document.createTextNode("Compare models "));
    analyzeButton.append(makeElement("span", "", "→"));
  }
}

function renderHistory(entries) {
  historyRows.replaceChildren();
  historyEmpty.hidden = entries.length > 0;
  for (const entry of entries) {
    const row = document.createElement("tr");
    row.append(makeElement("td", "", entry.text));
    row.append(makeElement("td", "", entry.model.replaceAll("_", " ")));
    row.append(makeElement("td", `history-verdict ${entry.verdict.toLowerCase()}`, entry.verdict));
    row.append(makeElement("td", "", `${(Number(entry.confidence) * 100).toFixed(1)}%`));
    row.append(makeElement("td", "", `${Number(entry.latency_ms).toFixed(2)} ms`));
    historyRows.append(row);
  }
}

async function loadHistory() {
  const response = await fetch("/history?limit=20");
  if (!response.ok) throw new Error("Could not load recent history.");
  const data = await response.json();
  renderHistory(data.history);
}

analyzeButton.addEventListener("click", analyze);
document.getElementById("refresh-history").addEventListener("click", () => {
  loadHistory().catch((error) => showError(error.message));
});

Promise.all([loadModels(), loadHistory()]).catch((error) => showError(error.message));

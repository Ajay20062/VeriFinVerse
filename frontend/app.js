(() => {
  "use strict";

  const DEFAULT_API = "http://localhost:8000";
  const MODEL_LABELS = {
    logistic_regression: "Logistic Regression",
    xgboost: "XGBoost",
    finbert: "FinBERT",
  };
  const apiIndicator = document.querySelector(".api-indicator");
  const connectionLabel = document.getElementById("connection-label");
  const modelOptions = document.getElementById("model-options");
  const modelStatusList = document.getElementById("model-status-list");
  const historyList = document.getElementById("history-list");
  const resultContent = document.getElementById("result-content");
  const analyzeButton = document.getElementById("analyze-button");
  const formFeedback = document.getElementById("form-feedback");
  const analysisForm = document.getElementById("analysis-form");
  const apiInput = document.getElementById("api-url");
  const mobileToggle = document.querySelector(".menu-toggle");
  const mobileNav = document.getElementById("mobile-nav");
  let baseUrl = normalizeBaseUrl(localStorage.getItem("verifinverse-api-url") || DEFAULT_API);
  let modelStatuses = [];
  let historyRecords = [];
  let selectedHistoryId = null;

  function normalizeBaseUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
  }

  function apiUrl(path) {
    return `${baseUrl}${path}`;
  }

  function setConnectionState(state, label) {
    apiIndicator.dataset.state = state;
    connectionLabel.textContent = label;
  }

  function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function modelLabel(name) {
    return MODEL_LABELS[name] || name.replace(/_/g, " ");
  }

  function prettyTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value || "Time unavailable");
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
  }

  function confidenceText(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "Not provided";
    return `${Math.round(value * 100)}% confidence`;
  }

  function predictionClass(label) {
    const normalized = String(label || "").toUpperCase();
    if (normalized === "FAKE") return "fake";
    if (normalized === "REAL") return "real";
    if (normalized === "UNCERTAIN") return "uncertain";
    return "";
  }

  function predictionBadge(label) {
    const badge = makeElement("span", `prediction-badge ${predictionClass(label)}`, label || "Unknown");
    return badge;
  }

  function setFeedback(message, kind) {
    formFeedback.textContent = message;
    formFeedback.dataset.kind = kind || "notice";
    formFeedback.hidden = !message;
  }

  function apiErrorMessage(payload, fallback) {
    const detail = payload && payload.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const message = detail.message || "The API could not complete this request.";
      const unavailable = Array.isArray(detail.unavailable_models) && detail.unavailable_models.length
        ? ` Unavailable: ${detail.unavailable_models.map(modelLabel).join(", ")}.`
        : "";
      return `${message}${unavailable}`;
    }
    return fallback;
  }

  async function requestJson(path, options) {
    const response = await fetch(apiUrl(path), options);
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`The API returned an unreadable response (${response.status}).`);
    }
    if (!response.ok) throw new Error(apiErrorMessage(payload, `Request failed (${response.status}).`));
    return payload;
  }

  function renderModelOptions() {
    modelOptions.replaceChildren();
    if (!modelStatuses.length) {
      modelOptions.append(makeElement("p", "inline-status", "Model availability has not loaded. You can still submit to receive the API's actual response."));
      return;
    }
    modelStatuses.forEach((model, index) => {
      const available = model.status === "ready";
      const label = makeElement("label", "model-option");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "models";
      checkbox.value = model.name;
      checkbox.disabled = !available;
      checkbox.checked = available;
      if (available && index === 0) checkbox.setAttribute("aria-label", `${modelLabel(model.name)} (available)`);
      label.append(checkbox, makeElement("span", "", modelLabel(model.name)));
      modelOptions.append(label);
    });
  }

  function renderModelStatuses() {
    modelStatusList.replaceChildren();
    if (!modelStatuses.length) {
      modelStatusList.append(makeElement("p", "inline-status", "Model status is currently unavailable."));
      document.getElementById("model-count").textContent = "— / 3";
      renderModelOptions();
      return;
    }
    const readyCount = modelStatuses.filter((item) => item.status === "ready").length;
    document.getElementById("model-count").textContent = `${readyCount} / ${modelStatuses.length}`;
    modelStatuses.forEach((model) => {
      const row = makeElement("div", "model-status");
      row.dataset.state = model.status || "unknown";
      const statusText = model.status === "ready" ? "Ready" : model.status === "error" ? "Load error" : "Not configured";
      row.append(
        makeElement("span", "model-led"),
        makeElement("span", "model-status-name", modelLabel(model.name)),
        makeElement("span", "model-status-state", statusText),
      );
      if (model.details) row.append(makeElement("span", "model-status-details", model.details));
      modelStatusList.append(row);
    });
    renderModelOptions();
  }

  function renderMetrics(records) {
    document.getElementById("metric-total").textContent = String(records.length);
    document.getElementById("metric-fake").textContent = String(records.filter((item) => String(item.prediction).toUpperCase() === "FAKE").length);
    document.getElementById("metric-real").textContent = String(records.filter((item) => String(item.prediction).toUpperCase() === "REAL").length);
  }

  function renderHistory() {
    historyList.replaceChildren();
    if (!historyRecords.length) {
      const empty = makeElement("div", "history-empty");
      empty.append(makeElement("span", "empty-glyph", "⌁"), makeElement("strong", "", "No analyses in history"), makeElement("p", "", "Saved results from this API will appear here."));
      historyList.append(empty);
      return;
    }
    historyRecords.forEach((record) => {
      const entry = makeElement("article", `history-entry${record.id === selectedHistoryId ? " is-selected" : ""}`);
      entry.tabIndex = 0;
      entry.setAttribute("role", "button");
      entry.setAttribute("aria-label", `View analysis ${record.id}: ${record.prediction}`);
      const textWrap = document.createElement("div");
      textWrap.append(makeElement("span", "history-entry-text", record.text));
      const meta = makeElement("span", "history-entry-meta", `#${record.id} · ${prettyTime(record.created_at)}`);
      textWrap.append(meta);
      entry.append(textWrap, predictionBadge(record.prediction));
      const activate = () => {
        selectedHistoryId = record.id;
        renderHistory();
        renderResult(record, true);
        document.getElementById("analysis-result").scrollIntoView({ behavior: "smooth", block: "nearest" });
      };
      entry.addEventListener("click", activate);
      entry.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
      historyList.append(entry);
    });
  }

  function renderResult(record, isHistory) {
    resultContent.className = "result-summary";
    resultContent.replaceChildren();
    const topline = makeElement("div", "result-topline");
    topline.append(predictionBadge(record.prediction));
    const confidence = makeElement("span", "result-confidence", confidenceText(record.confidence));
    topline.append(confidence);
    const timeLabel = makeElement("div", "result-time", `${isHistory ? `HISTORY #${record.id} · ` : `ANALYSIS #${record.id} · `}${prettyTime(record.created_at)}`);
    const story = makeElement("p", "result-story", record.text || "Text was not included in this response.");
    resultContent.append(topline, story, timeLabel);

    if (Array.isArray(record.models) && record.models.length) {
      const predictions = makeElement("div", "prediction-list");
      record.models.forEach((item) => {
        const row = makeElement("div", "prediction-row");
        row.append(
          makeElement("span", "prediction-model", modelLabel(item.model)),
          makeElement("span", "prediction-label", item.label || "Unknown"),
          makeElement("span", "prediction-confidence", typeof item.confidence === "number" ? confidenceText(item.confidence) : "No confidence"),
        );
        predictions.append(row);
        if (item.explanation && Object.keys(item.explanation).length) {
          const disclosure = document.createElement("details");
          const summary = makeElement("summary", "prediction-model", `${modelLabel(item.model)} explanation`);
          const explanation = makeElement("pre", "explanation-json", JSON.stringify(item.explanation, null, 2));
          disclosure.append(summary, explanation);
          predictions.append(disclosure);
        }
      });
      resultContent.append(predictions);
    } else {
      resultContent.append(makeElement("p", "inline-status", "The API did not include per-model details."));
    }
  }

  async function loadModels() {
    try {
      const response = await requestJson("/api/models");
      modelStatuses = Array.isArray(response.models) ? response.models : [];
      renderModelStatuses();
      setConnectionState("connected", "API available");
    } catch (error) {
      modelStatuses = [];
      renderModelStatuses();
      setConnectionState("error", "API unavailable");
      document.getElementById("connection-feedback").textContent = error.message;
    }
  }

  async function loadHistory() {
    historyList.replaceChildren(makeElement("div", "history-empty", "Loading saved records…"));
    try {
      const records = await requestJson("/api/history?limit=20");
      historyRecords = Array.isArray(records) ? records : [];
      renderMetrics(historyRecords);
      renderHistory();
      setConnectionState("connected", "API available");
    } catch (error) {
      historyRecords = [];
      renderMetrics([]);
      historyList.replaceChildren();
      const empty = makeElement("div", "history-empty");
      empty.append(makeElement("strong", "", "History could not be loaded"), makeElement("p", "", error.message));
      historyList.append(empty);
      setConnectionState("error", "API unavailable");
    }
  }

  analysisForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = document.getElementById("news-text").value.trim();
    if (!text) {
      setFeedback("Enter a headline or article before submitting.", "error");
      return;
    }
    const selectedModels = Array.from(analysisForm.querySelectorAll('input[name="models"]:checked')).map((input) => input.value);
    const requestBody = { text };
    if (selectedModels.length) requestBody.models = selectedModels;
    analyzeButton.disabled = true;
    analyzeButton.querySelector("span:last-child").textContent = "Analyzing…";
    setFeedback("", "");
    try {
      const result = await requestJson("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      renderResult({ ...result, text }, false);
      selectedHistoryId = result.id;
      setFeedback("Analysis returned by the connected API and saved to history.", "success");
      await loadHistory();
      const savedRecord = historyRecords.find((item) => item.id === result.id);
      if (savedRecord) {
        selectedHistoryId = savedRecord.id;
        renderHistory();
      }
      setConnectionState("connected", "API available");
    } catch (error) {
      setFeedback(error.message, "error");
      if (/failed to fetch|networkerror|load failed/i.test(error.message)) setConnectionState("error", "API unavailable");
    } finally {
      analyzeButton.disabled = false;
      analyzeButton.querySelector("span:last-child").textContent = "Analyze story";
    }
  });

  document.getElementById("news-text").addEventListener("input", (event) => {
    document.getElementById("character-count").textContent = `${event.target.value.length.toLocaleString()} / 20,000`;
  });

  document.getElementById("refresh-history").addEventListener("click", () => loadHistory());

  document.getElementById("connection-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const nextUrl = normalizeBaseUrl(apiInput.value);
    if (!nextUrl) return;
    try {
      const parsed = new URL(nextUrl);
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("Use an http or https API URL.");
    } catch (error) {
      document.getElementById("connection-feedback").textContent = error.message || "Enter a valid API base URL.";
      return;
    }
    baseUrl = nextUrl;
    localStorage.setItem("verifinverse-api-url", baseUrl);
    document.getElementById("connection-feedback").textContent = "Connection URL saved. Checking service…";
    updateApiDocsLink();
    await Promise.all([loadModels(), loadHistory()]);
    document.getElementById("connection-feedback").textContent = "Connection checked.";
  });

  function updateApiDocsLink() {
    const link = document.getElementById("api-doc-link");
    link.href = `${baseUrl}/docs`;
  }

  mobileToggle.addEventListener("click", () => {
    const expanded = mobileToggle.getAttribute("aria-expanded") === "true";
    mobileToggle.setAttribute("aria-expanded", String(!expanded));
    mobileNav.hidden = expanded;
  });

  mobileNav.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      mobileNav.hidden = true;
      mobileToggle.setAttribute("aria-expanded", "false");
    }
  });

  document.querySelectorAll(".primary-nav .nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".primary-nav .nav-link").forEach((item) => item.classList.remove("is-active"));
      link.classList.add("is-active");
    });
  });

  apiInput.value = baseUrl;
  updateApiDocsLink();
  Promise.all([loadModels(), loadHistory()]);
})();

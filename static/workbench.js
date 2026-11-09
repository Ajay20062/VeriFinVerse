const algorithmSelect = document.getElementById("algorithm-select");
const algorithmDescription = document.getElementById("algorithm-description");
const algorithmCatalog = document.getElementById("algorithm-catalog");
const datasetStats = document.getElementById("dataset-stats");
const datasetError = document.getElementById("dataset-error");
const algorithmError = document.getElementById("algorithm-error");
const resultOutput = document.getElementById("algorithm-result");
const runButton = document.getElementById("run-algorithm");

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function readableName(name) {
  return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function showError(target, error) {
  target.textContent = error;
  target.hidden = false;
}

async function loadDatasetSummary() {
  datasetError.hidden = true;
  const response = await fetch("/dataset/summary");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load the dataset.");
  datasetStats.replaceChildren();
  const values = [
    ["Claims", data.rows],
    ["REAL", data.label_counts.REAL],
    ["FAKE", data.label_counts.FAKE],
    ["Mean words", data.mean_words.toFixed(1)],
    ["Median words", data.median_words],
    ["Mean characters", data.mean_characters.toFixed(1)],
  ];
  for (const [label, value] of values) {
    const card = node("div", "stat-card");
    card.append(node("span", "", label), node("strong", "", String(value)));
    datasetStats.append(card);
  }
}

async function loadAlgorithms() {
  const response = await fetch("/algorithms");
  const data = await response.json();
  if (!response.ok) throw new Error("Could not load the algorithm catalog.");
  algorithmSelect.replaceChildren();
  algorithmCatalog.replaceChildren();
  for (const [name, description] of Object.entries(data.algorithms)) {
    const option = node("option", "", readableName(name));
    option.value = name;
    algorithmSelect.append(option);
    const item = node("li");
    item.append(node("strong", "", readableName(name)), node("span", "", description));
    algorithmCatalog.append(item);
  }
  updateDescription(data.algorithms);
  algorithmSelect.addEventListener("change", () => updateDescription(data.algorithms));
}

function updateDescription(catalog) {
  algorithmDescription.textContent = catalog[algorithmSelect.value] || "";
}

async function runAlgorithm() {
  algorithmError.hidden = true;
  resultOutput.textContent = "";
  runButton.disabled = true;
  runButton.textContent = "Running...";
  try {
    const response = await fetch("/algorithms/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm: algorithmSelect.value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Algorithm run failed.");
    resultOutput.textContent = JSON.stringify(data.result, null, 2);
  } catch (error) {
    showError(algorithmError, error.message);
  } finally {
    runButton.disabled = false;
    runButton.replaceChildren(document.createTextNode("Run algorithm "));
    runButton.append(node("span", "", "→"));
  }
}

document.getElementById("refresh-dataset").addEventListener("click", () => {
  loadDatasetSummary().catch((error) => showError(datasetError, error.message));
});
runButton.addEventListener("click", runAlgorithm);

Promise.all([loadAlgorithms(), loadDatasetSummary()]).catch((error) => {
  showError(datasetError, error.message);
});

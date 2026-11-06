
const API = "";
let selectedCSV = null;
let modelFeatures = ["Time", "Amount", ...Array.from({length: 28}, (_, i) => `V${i + 1}`)];

const $ = (id) => document.getElementById(id);

function showPage(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-link").forEach(b => b.classList.remove("active"));

  const page = $(`page-${name}`);
  if (page) page.classList.add("active");

  const nav = document.querySelector(`[data-page="${name}"]`);
  if (nav) nav.classList.add("active");

  const labels = {
    overview: "Overview",
    scanner: "Fraud Scanner",
    batch: "Batch Intelligence",
    history: "Transaction History",
    performance: "Model Performance",
    dataset: "Dataset Insights",
    settings: "Settings"
  };

  $("currentSection").textContent = labels[name] || "Overview";

  if (name === "history") renderHistory();
  if (name === "performance") loadMetrics();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(window.toastTimer);
  window.toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

function number(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function percent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return (n <= 1 ? n * 100 : n).toFixed(2) + "%";
}

async function getJSON(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function postJSON(path, data) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

// Build all 28 PCA inputs.
// Generate all 28 editable PCA feature inputs.
function buildPCAInputs() {
    const grid = $("pcaGrid");
    if (!grid) return;

    grid.innerHTML = "";

    for (let i = 1; i <= 28; i++) {
        const label = document.createElement("label");
        label.textContent = `V${i}`;

        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = "0";
        input.id = `v${i}`;
        input.disabled = false;
        input.readOnly = false;
        input.placeholder = "Enter V" + i;

        label.appendChild(input);
        grid.appendChild(label);
    }

    // Show editable features by default.
    grid.classList.add("open");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns =
        "repeat(auto-fit, minmax(100px, 1fr))";

    const button = document.querySelector(".collapse-button");
    if (button) button.textContent = "HIDE FEATURES ⌃";
}

function toggleFeatures() {
    const grid = $("pcaGrid");
    const button = document.querySelector(".collapse-button");

    if (!grid) return;

    const isOpen = grid.classList.toggle("open");

    grid.style.display = isOpen ? "grid" : "none";

    if (button) {
        button.textContent = isOpen
            ? "HIDE FEATURES ⌃"
            : "EDIT FEATURES ⌄";
    }
}
function getTransactionFeatures() {
  const features = {
    Time: Number($("time").value || 0),
    Amount: Number($("amount").value || 0)
  };

  for (let i = 1; i <= 28; i++) {
    features[`V${i}`] = Number($(`v${i}`).value || 0);
  }

  // Include only the features expected by the saved model.
  const filtered = {};
  modelFeatures.forEach(name => {
    filtered[name] = features[name] ?? 0;
  });
  return filtered;
}

async function analyzeTransaction() {
  const button = $("analyzeButton");
  const error = $("scannerError");
  error.textContent = "";

  if ($("amount").value === "" || Number($("amount").value) < 0) {
    error.textContent = "Please enter a valid transaction amount.";
    return;
  }

  if ($("time").value !== "" && Number($("time").value) < 0) {
    error.textContent = "Transaction time cannot be negative.";
    return;
  }

  button.disabled = true;
  button.textContent = "Analyzing transaction...";

  try {
    const features = getTransactionFeatures();
    const result = await postJSON("/api/predict", {features});

    displayResult(result);
    saveHistory({
      date: new Date().toLocaleString(),
      amount: Number($("amount").value),
      prediction: result.prediction,
      probability: result.fraud_probability,
      risk: result.risk_level
    });

    toast("Transaction analysis completed.");
  } catch (err) {
    error.textContent = err.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "✦ &nbsp; Analyze transaction <span>↗</span>";
  }
}

function displayResult(result) {
  $("resultEmpty").classList.add("hidden");
  $("resultOutput").classList.remove("hidden");

  const fraud = result.prediction === "Fraudulent";
  const probability = Number(result.fraud_probability);

  $("resultStatus").textContent = fraud ? "⚠ FRAUDULENT TRANSACTION" : "✓ GENUINE TRANSACTION";
  $("resultStatus").classList.toggle("danger", fraud);
  $("riskNumber").innerHTML = `${number(probability)}<small>%</small>`;
  $("riskLevel").textContent = result.risk_level;
  $("modelDecision").textContent = result.prediction.toUpperCase();
  $("riskTrack").style.width = `${Math.max(0, Math.min(100, probability))}%`;

  $("resultExplanation").textContent = fraud
    ? "The trained model classified this transaction as potentially fraudulent. Review the transaction and verify it using additional evidence before taking action."
    : "The trained model classified this transaction as likely genuine. This is a model prediction, not a guarantee that the transaction is safe.";
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem("fraudshield_history") || "[]");
  } catch {
    return [];
  }
}

function saveHistory(item) {
  const history = readHistory();
  history.unshift(item);
  localStorage.setItem("fraudshield_history", JSON.stringify(history.slice(0, 100)));
}

function renderHistory() {
  const history = readHistory();
  $("historyCount").textContent = `${history.length} RECORDS`;

  if (!history.length) {
    $("historyRows").innerHTML = `<tr><td colspan="5">No analyses yet. Visit Fraud Scanner to begin.</td></tr>`;
    return;
  }

  $("historyRows").innerHTML = history.map(item => `
    <tr>
      <td>${escapeHTML(item.date)}</td>
      <td>₹${number(item.amount)}</td>
      <td><span class="status-pill ${item.prediction === "Fraudulent" ? "bad" : "good"}">${escapeHTML(item.prediction)}</span></td>
      <td>${number(item.probability)}%</td>
      <td>${escapeHTML(item.risk)}</td>
    </tr>
  `).join("");
}

function clearHistory() {
  if (!confirm("Clear all locally saved transaction history?")) return;
  localStorage.removeItem("fraudshield_history");
  renderHistory();
  toast("History cleared.");
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;"
  })[char]);
}

// CSV parser supports quoted values and commas inside quoted fields.
function parseCSV(text) {
  const rows = [];
  let row = [], value = "", quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && quoted && next === '"') {
      value += '"';
      i++;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(value.trim());
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i++;
      row.push(value.trim());
      if (row.some(cell => cell !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  row.push(value.trim());
  if (row.some(cell => cell !== "")) rows.push(row);
  return rows;
}

$("csvFile").addEventListener("change", event => {
  selectedCSV = event.target.files[0] || null;
  $("fileName").textContent = selectedCSV ? selectedCSV.name : "No file selected";
  $("batchError").textContent = "";
});

async function runBatch() {
  $("batchError").textContent = "";

  if (!selectedCSV) {
    $("batchError").textContent = "Please choose a CSV file first.";
    return;
  }

  const button = $("batchButton");
  button.disabled = true;
  button.textContent = "Analyzing dataset...";

  try {
    const raw = await selectedCSV.text();
    const parsed = parseCSV(raw);

    if (parsed.length < 2) throw new Error("The CSV file has no transaction rows.");

    const headers = parsed[0].map(h => h.trim());
    const headerMap = {};
    headers.forEach(h => headerMap[h.toLowerCase()] = h);

    const missing = modelFeatures.filter(f => !headerMap[f.toLowerCase()]);
    if (missing.length) {
      throw new Error("Missing required columns: " + missing.join(", "));
    }

    const transactions = parsed.slice(1).filter(row => row.some(v => v !== "")).map((row, index) => {
      const values = {};
      modelFeatures.forEach(feature => {
        const columnIndex = headers.findIndex(h => h.toLowerCase() === feature.toLowerCase());
        const rawValue = row[columnIndex];

        if (rawValue === undefined || rawValue.trim() === "" || !Number.isFinite(Number(rawValue))) {
          throw new Error(`Invalid value for ${feature} on CSV row ${index + 2}.`);
        }

        values[feature] = Number(rawValue);
      });
      return values;
    });

    if (!transactions.length) throw new Error("No valid transaction rows found.");
    if (transactions.length > 10000) throw new Error("Maximum 10,000 rows per upload.");

    const result = await postJSON("/api/batch", transactions);

    $("batchStats").classList.remove("hidden");
    $("batchTableCard").classList.remove("hidden");

    $("batchTotal").textContent = result.total.toLocaleString();
    $("batchFraud").textContent = result.fraud_count.toLocaleString();
    $("batchGenuine").textContent = result.genuine_count.toLocaleString();
    $("batchRate").textContent = result.total
      ? (result.fraud_count / result.total * 100).toFixed(2) + "%"
      : "0%";

    $("batchRows").innerHTML = result.results.map(item => `
      <tr>
        <td>#${item.transaction_number}</td>
        <td><span class="status-pill ${item.prediction === "Fraudulent" ? "bad" : "good"}">${item.prediction}</span></td>
        <td>${number(item.fraud_probability)}%</td>
        <td>${item.risk_level}</td>
      </tr>
    `).join("");

    toast(`Analyzed ${result.total} transactions.`);
  } catch (err) {
    $("batchError").textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = "Analyze batch ↗";
  }
}

// Read metrics with support for common JSON metric layouts.
function findMetric(source, aliases) {
  if (!source || typeof source !== "object") return null;

  const normalized = aliases.map(a => a.toLowerCase().replace(/[^a-z0-9]/g, ""));
  for (const [key, value] of Object.entries(source)) {
    const k = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    if (normalized.includes(k) && typeof value === "number") return value;
  }

  for (const value of Object.values(source)) {
    if (value && typeof value === "object") {
      const found = findMetric(value, aliases);
      if (found !== null) return found;
    }
  }
  return null;
}

function setMetric(id, value) {
  if (value !== null && value !== undefined) $(id).textContent = percent(value);
}
    async function loadMetrics() {
    try {
        const data = await getJSON("/api/metrics");

        if (!data) return;

        const precision = findMetric(data, ["precision"]);
        const recall = findMetric(data, ["recall"]);
        const prAuc = findMetric(data, ["pr_auc", "average_precision", "prauc"]);
        const rocAuc = findMetric(data, ["roc_auc", "rocauc"]);
        const f1 = findMetric(data, ["f1", "f1_score"]);

        setMetric("perfPR", prAuc);
        setMetric("perfROC", rocAuc);
        setMetric("perfPrecision", precision);
        setMetric("perfRecall", recall);

        setMetric("precisionValue", precision);
        setMetric("recallValue", recall);
        setMetric("prAuc", prAuc);
        setMetric("rocAuc", rocAuc);
        setMetric("f1Value", f1);

        renderModelBars(data);

    } catch (err) {
        console.warn("Metrics could not be loaded:", err.message);
    }
}

function renderModelBars(data) {
  const container = $("modelBars");
  if (!container) return;

  const models = data && typeof data === "object" ? data : {};
  const entries = [];

  function walk(obj) {
    if (!obj || typeof obj !== "object") return;
    for (const [key, value] of Object.entries(obj)) {
      if (value && typeof value === "object") {
        const score = findMetric(value, ["pr_auc", "average_precision", "prauc"]);
        if (score !== null) entries.push({name: key, score});
        else walk(value);
      }
    }
  }
  walk(models);

  if (!entries.length) {
    container.innerHTML = `<div class="legend-note">Training comparison metrics are not available in the saved metrics file.</div>`;
    return;
  }

  container.innerHTML = entries.map(item => {
    const value = item.score <= 1 ? item.score * 100 : item.score;
    return `<div class="bar-row">
      <span>${escapeHTML(item.name)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, value))}%"></div></div>
      <b>${value.toFixed(2)}%</b>
    </div>`;
  }).join("");
}

async function refreshDashboard() {
  toast("Refreshing platform information...");
  await Promise.all([checkConnection(), loadMetrics()]);
  toast("Dashboard refreshed.");
}

async function checkConnection() {
  try {
    const [health, info] = await Promise.all([
      getJSON("/api/health"),
      getJSON("/api/model-info")
    ]);

    $("sideStatus").textContent = "Engine operational";
    $("settingsStatus").textContent = "Connected";
    $("settingsStatus").style.color = "#63e5b4";
    $("settingsModel").textContent = health.model;
    $("modelName").textContent = health.model;
    $("settingsFeatures").textContent = info.features.length;
    modelFeatures = info.features;
  } catch (err) {
    $("sideStatus").textContent = "Connection unavailable";
    $("settingsStatus").textContent = "Offline";
    $("settingsStatus").style.color = "#ff729f";
    console.error(err);
  }
}

async function initialize() {
  buildPCAInputs();
  renderHistory();
  await checkConnection();
  await loadMetrics();
}

document.addEventListener("DOMContentLoaded", async () => {
    const authScreen = $("authScreen");
    const appShell = document.querySelector(".app-shell");

    if ("scrollRestoration" in history) {
        history.scrollRestoration = "manual";
    }

    // Initially hide dashboard and login screen
    authScreen.hidden = true;
    appShell.style.display = "none";

    try {
        const response = await fetch("/api/auth/me", {
            credentials: "same-origin"
        });

        if (response.ok) {
            await loadUserProfile();

            authScreen.hidden = true;
            appShell.style.display = "flex";

            await initialize();
        } else {
            appShell.style.display = "none";
            authScreen.hidden = false;
            authScreen.style.display = "grid";
            $("authError").textContent = "";
        }

    } catch (error) {
        appShell.style.display = "none";
        authScreen.hidden = false;
        authScreen.style.display = "grid";
        $("authError").textContent = "Unable to connect to the server.";
    }

    window.scrollTo(0, 0);
});

// ===== AUTHENTICATION FRONTEND =====

let authMode = "login";

function switchAuth(mode) {
  authMode = mode;

  $("authName").style.display = mode === "register" ? "block" : "none";
  $("authName").required = mode === "register";

  $("authTitle").textContent =
    mode === "login" ? "Welcome Back!" : "Create Your Account";

  $("authSubtitle").textContent =
    mode === "login"
      ? "Sign in to access your FraudShield dashboard."
      : "Register to start protecting transactions.";

  $("authSubmit").textContent =
    mode === "login" ? "Login →" : "Create Account →";

  $("loginTab").classList.toggle("selected", mode === "login");
  $("registerTab").classList.toggle("selected", mode === "register");
  $("authError").textContent = "";
}
$("authScreen").hidden = true;
$("authScreen").style.setProperty("display", "none", "important");

document.querySelector(".app-shell").style.display = "flex";

console.log("AUTH DISPLAY:", getComputedStyle($("authScreen")).display);
console.log("APP DISPLAY:", getComputedStyle(document.querySelector(".app-shell")).display);
$("authForm").addEventListener("submit", async function(event) {
  event.preventDefault();

  const data = {
    email: $("authEmail").value.trim(),
    password: $("authPassword").value
};

if (authMode === "login") {
    data.role = $("authRole").value;
} else {
    data.name = $("authName").value.trim();
}

  try {
    const response = await fetch(
      authMode === "login"
        ? "/api/auth/login"
        : "/api/auth/register",
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        credentials: "same-origin",
        body: JSON.stringify(data)
      }
    );

    const result = await response.json();

    if (!response.ok) throw new Error(result.detail || "Authentication failed.");

    await loadUserProfile();

$("authScreen").hidden = true;
document.querySelector(".app-shell").style.display = "flex";

window.scrollTo(0, 0);

await initialize();

  } catch (error) {
    $("authError").textContent = error.message;
  }
});


async function loadUserProfile() {
    try {
        const response = await fetch("/api/auth/me", {
            credentials: "same-origin"
        });

        if (!response.ok) {
            document.querySelector(".app-shell").style.display = "none";
            $("authScreen").style.display = "flex";
            return;
        }

        const user = await response.json();

        const name = user.name || "User";
        const email = user.email || "";
        const role = user.role || "NEW USER";

        const initials = name
            .trim()
            .split(/\s+/)
            .map(part => part.charAt(0))
            .join("")
            .substring(0, 2)
            .toUpperCase();

        const setText = (id, value) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        };

        setText("profileName", name);
        setText("profileRole", role);
        setText("userAvatar", initials);
        setText("dropdownName", name);
        setText("dropdownEmail", email);
        setText("dropdownRole", role);
        setText("dropdownAvatar", initials);
        setText("topAvatar", initials);
        setText("dashboardUserName", name + ".");

    } catch (error) {
        console.error("Profile loading failed:", error);
    }
}


async function logoutFraudShield() {
    try {
        await fetch("/api/auth/logout", {
            method: "POST",
            credentials: "same-origin"
        });
    } catch (error) {
        console.error("Logout error:", error);
    }

    // Prevent browser from restoring the old scroll position
    if ("scrollRestoration" in history) {
        history.scrollRestoration = "manual";
    }

    window.scrollTo(0, 0);
    window.location.reload();
}

/* ==========================================
   FRAUDSHIELD - USER PROFILE & LOGOUT
========================================== */

async function loadAnimatedProfile() {
    try {
        const response = await fetch("/api/auth/me", {
            credentials: "same-origin"
        });

        if (!response.ok) {
            return;
        }

        const user = await response.json();

        const name = user.name || "User";
        const email = user.email || "";
        const role = user.role || "NEW USER";

        const initials = name
            .trim()
            .split(/\s+/)
            .map(part => part[0])
            .join("")
            .substring(0, 2)
            .toUpperCase();

        const update = (id, value) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        };

        update("profileName", name);
        update("userAvatar", initials);
        update("profileRole", role);

        update("dropdownName", name);
        update("dropdownAvatar", initials);
        update("dropdownEmail", email);
        update("dropdownRole", role);

        const topAvatar = document.querySelector(".top-avatar");
        if (topAvatar) topAvatar.textContent = initials;

    } catch (error) {
        console.error("Profile loading failed:", error);
    }
}

function initializeRoleSelection() {
    const buttons = document.querySelectorAll(".role-option");
    const roleSelect = document.getElementById("authRole");

    buttons.forEach(button => {
        button.addEventListener("click", () => {
            buttons.forEach(item => {
                item.classList.remove("selected");
            });

            button.classList.add("selected");

            if (roleSelect) {
                roleSelect.value = button.dataset.role;
            }
        });
    });

    const selectedButton = document.querySelector(
        ".role-option.selected"
    );

    if (selectedButton && roleSelect) {
        roleSelect.value = selectedButton.dataset.role;
    }
}
function initializeAnimatedProfile() {
    const card = document.getElementById("userProfileCard");
    const toggle = document.getElementById("profileToggle");
    const logout = document.getElementById("profileLogout");

    if (!card || !toggle || !logout) return;

    toggle.addEventListener("click", () => {
        card.classList.toggle("open");
    });

    document.addEventListener("click", event => {
        if (!card.contains(event.target)) {
            card.classList.remove("open");
        }
    });

    logout.addEventListener("click", logoutFraudShield);

    loadAnimatedProfile();
}

async function logoutFraudShield() {
    const button = document.getElementById("profileLogout");

    if (button) {
        button.disabled = true;
        button.textContent = "Signing out...";
    }

    try {
        const response = await fetch("/api/auth/logout", {
            method: "POST",
            credentials: "same-origin"
        });

        if (!response.ok) {
            throw new Error("Logout failed");
        }

        window.location.reload();

    } catch (error) {
        console.error(error);
        if (button) {
            button.disabled = false;
            button.textContent = "Sign Out";
        }
        alert("Unable to sign out. Please try again.");
    }
}

document.addEventListener(
    "DOMContentLoaded",
    initializeAnimatedProfile
);
async function updateUserProfile() {
    try {
        const response = await fetch("/api/auth/me", {
            credentials: "same-origin"
        });

        if (!response.ok) {
            console.error("User authentication failed:", response.status);
            return;
        }

        const user = await response.json();
        const name = user.name || "User";
        const initials = name.trim()
            .split(/\s+/)
            .map(word => word[0])
            .join("")
            .substring(0, 2)
            .toUpperCase();

        document.getElementById("topAvatar").textContent = initials;
        document.getElementById("dashboardUserName").textContent = name + ".";
    } catch (error) {
        console.error("Profile error:", error);
    }
}

document.addEventListener("DOMContentLoaded", updateUserProfile);

/* ===== FRAUDSHIELD ROLE SELECTOR ===== */

document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".role-option");
    const roleSelect = document.getElementById("authRole");

    if (!roleSelect || !buttons.length) return;

    buttons.forEach(button => {
        button.addEventListener("click", () => {
            buttons.forEach(item => item.classList.remove("selected"));

            button.classList.add("selected");
            roleSelect.value = button.dataset.role;
        });
    });
});
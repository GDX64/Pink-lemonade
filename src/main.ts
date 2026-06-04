import { runApplicationPage } from "./pages/application-page";
import { runBenchmarkPage } from "./pages/benchmark-page";
import { runFigurePage } from "./pages/figures-page";
import { runMsePage } from "./pages/mse-page";

const root = document.createElement("div");
root.style.minHeight = "100vh";
root.style.fontFamily = "ui-sans-serif, Segoe UI, Helvetica, Arial, sans-serif";
root.style.background = "linear-gradient(135deg, #f7fbff 0%, #f4f8ee 100%)";
root.style.color = "#16324f";
root.style.padding = "24px";
root.style.boxSizing = "border-box";

const title = document.createElement("h1");
title.textContent = "Reproducibility Runner";
title.style.margin = "0 0 12px 0";

const subtitle = document.createElement("p");
subtitle.textContent =
  "Run paper artifacts in-browser: application, merge benchmark, MSE estimation, and figure rendering.";
subtitle.style.margin = "0 0 20px 0";

const menu = document.createElement("ul");
menu.style.display = "flex";
menu.style.gap = "10px";
menu.style.padding = "0";
menu.style.margin = "0 0 20px 0";
menu.style.listStyle = "none";
menu.style.flexWrap = "wrap";

const output = document.createElement("div");
output.style.background = "#ffffff";
output.style.border = "1px solid #d8e2ee";
output.style.borderRadius = "12px";
output.style.padding = "16px";

const appMount = document.createElement("div");
appMount.style.minHeight = "620px";

let activeAction = 0;

const actions: Array<{
  page: string;
  label: string;
  run: () => Promise<void>;
}> = [
  {
    page: "application",
    label: "application",
    run: async () => {
      await runApplicationPage({ output, appMount, setInfo });
    },
  },
  {
    page: "benchmark",
    label: "benchmark",
    run: async () => {
      setInfo("Running merge benchmark in browser...");
      clearOutput();
      const table = await runBenchmarkPage();
      output.appendChild(renderTable(table));
      setInfo("Benchmark completed.");
    },
  },
  {
    page: "mse",
    label: "Mean Squared Error",
    run: async () => {
      setInfo("Running Monte Carlo MSE estimation...");
      clearOutput();
      const table = await runMsePage((progress) => {
        const overallPct = (progress.overallProgress * 100).toFixed(1);
        const scenarioPct = (progress.scenarioProgress * 100).toFixed(1);
        setInfo(
          `MSE estimation ${overallPct}% (N=${progress.n.toLocaleString("en-US")}, ${progress.acceptedSamples}/${progress.targetSamples}, scenario ${scenarioPct}%)`,
        );
      });
      output.appendChild(renderTable(table));
      setInfo("MSE estimation completed.");
    },
  },
  {
    page: "figure-1",
    label: "figure-1",
    run: async () => {
      await runFigurePage("figure-1", { output, appMount, setInfo });
    },
  },
  {
    page: "figure-2",
    label: "figure-2",
    run: async () => {
      await runFigurePage("figure-2", { output, appMount, setInfo });
    },
  },
  {
    page: "figure-3",
    label: "figure-3",
    run: async () => {
      await runFigurePage("figure-3", { output, appMount, setInfo });
    },
  },
];

for (const [idx, action] of actions.entries()) {
  const li = document.createElement("li");
  const button = document.createElement("button");
  button.textContent = action.label;
  button.style.border = "1px solid #97acc2";
  button.style.borderRadius = "10px";
  button.style.padding = "10px 14px";
  button.style.background = "#f0f6fd";
  button.style.color = "#12304d";
  button.style.fontWeight = "600";
  button.style.cursor = "pointer";
  button.onclick = () => {
    navigateToPage(action.page);
  };
  li.appendChild(button);
  menu.appendChild(li);
}

const info = document.createElement("p");
info.style.margin = "0 0 10px 0";
info.style.fontWeight = "600";

document.body.style.margin = "0";
document.body.style.overflow = "auto";
document.body.appendChild(root);

const pageFromQuery = new URLSearchParams(window.location.search)
  .get("page")
  ?.toLowerCase();
const selectedIndex = pageFromQuery
  ? actions.findIndex((a) => a.page === pageFromQuery)
  : -1;
const isHome = selectedIndex < 0;

if (isHome) {
  root.style.padding = "24px";
  root.appendChild(title);
  root.appendChild(subtitle);
  root.appendChild(menu);
  root.appendChild(info);
  root.appendChild(output);
  setInfo("Choose a page to run.");
  clearOutput();
  const helper = document.createElement("p");
  helper.textContent =
    "Home page. Selecting an option navigates to ?page=<name> and runs that page.";
  output.appendChild(helper);
  updateMenuHighlight();
} else {
  activeAction = selectedIndex;
  root.style.padding = "0";
  output.style.background = "transparent";
  output.style.border = "none";
  output.style.borderRadius = "0";
  output.style.padding = "0";
  output.style.minHeight = "100vh";
  root.appendChild(output);
  void actions[activeAction]!.run();
}

function setInfo(text: string) {
  info.textContent = text;
}

function clearOutput() {
  output.innerHTML = "";
}

function updateMenuHighlight() {
  if (!menu.isConnected) return;
  for (const [i, li] of Array.from(menu.children).entries()) {
    const button = li.firstElementChild as HTMLButtonElement;
    button.style.background = "#f0f6fd";
    button.style.borderColor = "#97acc2";
  }
}

function navigateToPage(page: string) {
  const target = new URL(window.location.origin);
  target.searchParams.set("page", page);
  window.location.assign(target.toString());
}

function renderTable(rows: Array<Record<string, string>>) {
  const table = document.createElement("table");
  table.style.borderCollapse = "collapse";
  table.style.width = "100%";
  if (rows.length === 0) return table;

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const key of Object.keys(rows[0]!)) {
    const th = document.createElement("th");
    th.textContent = key;
    th.style.border = "1px solid #d8e2ee";
    th.style.padding = "8px";
    th.style.textAlign = "left";
    th.style.background = "#f6fbff";
    headRow.appendChild(th);
  }
  head.appendChild(headRow);
  table.appendChild(head);

  const body = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of Object.values(row)) {
      const td = document.createElement("td");
      td.textContent = value;
      td.style.border = "1px solid #d8e2ee";
      td.style.padding = "8px";
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  table.appendChild(body);
  return table;
}


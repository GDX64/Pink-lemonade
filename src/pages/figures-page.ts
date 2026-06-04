import { renderFigureChart } from "../examples/rasterizing/rasterizing";

export async function runFigurePage(
  kind: "figure-1" | "figure-2" | "figure-3",
  args: {
    output: HTMLElement;
    appMount: HTMLElement;
    setInfo: (text: string) => void;
  },
) {
  const { output, appMount, setInfo } = args;
  setInfo(
    kind === "figure-1"
      ? "Rendered Figure 1 (full KDE reference) at 800x600."
      : kind === "figure-2"
        ? "Rendered Figure 2 (merged approximation) at 800x600."
        : "Rendered Figure 3 (merged approximation + selected area + zoom) at 800x600.",
  );
  output.innerHTML = "";
  output.style.margin = "0";
  output.style.padding = "0";
  output.style.width = "100vw";
  output.style.height = "100vh";

  const figureHost = appMount;
  figureHost.innerHTML = "";
  figureHost.style.width = "100vw";
  figureHost.style.height = "100vh";
  figureHost.style.minHeight = "100vh";
  figureHost.style.display = "flex";
  figureHost.style.alignItems = "center";
  figureHost.style.justifyContent = "center";

  output.appendChild(figureHost);
  await renderFigureChart(kind, figureHost);
}

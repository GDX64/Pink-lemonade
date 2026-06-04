import { rasterizingApplication } from "../examples/rasterizing/rasterizing";

export async function runApplicationPage(args: {
  output: HTMLElement;
  appMount: HTMLElement;
  setInfo: (text: string) => void;
}) {
  const { output, appMount, setInfo } = args;
  setInfo("Launching interactive example page...");
  output.innerHTML = "";
  output.style.margin = "0";
  output.style.padding = "0";
  output.style.width = "100vw";
  output.style.height = "100vh";
  output.appendChild(appMount);
  appMount.innerHTML = "";
  appMount.style.width = "100vw";
  appMount.style.height = "100vh";
  appMount.style.minHeight = "100vh";
  await rasterizingApplication(appMount);
  setInfo("Application loaded.");
}

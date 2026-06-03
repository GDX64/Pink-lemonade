// import { resolve } from "node:path";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    // lib: {
    //   entry: resolve(__dirname, "src/index.ts"),
    //   name: "PinkLemonade",
    //   formats: ["es"],
    //   fileName: "index",
    // },
    sourcemap: false,
  },
});

// import { resolve } from "node:path";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const singleFile = process.env.SINGLE_FILE === "true";

export default defineConfig({
  plugins: [singleFile ? viteSingleFile() : undefined].filter(Boolean),
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

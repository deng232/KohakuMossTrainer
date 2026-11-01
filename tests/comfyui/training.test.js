import assert from "node:assert/strict";
import test from "node:test";
import { migrateTargetArea, uploadDataset } from "../../web/training.js";

test("folder upload preserves image/caption pairing and skips unrelated files", async () => {
  const calls = [];
  const api = { async fetchApi(url, options) {
    calls.push([url, options]);
    return new Response(JSON.stringify(url.endsWith("/datasets")
      ? { upload_id: "id", path: "/server/input/dataset", extensions: [".png", ".txt"] }
      : { ok: true }));
  } };
  const files = ["dataset/2_style/a.png", "dataset/2_style/a.txt", "dataset/code.py"].map((path) => ({
    webkitRelativePath: path, name: path.split("/").at(-1),
  }));
  assert.equal(await uploadDataset(files, api), "/server/input/dataset");
  assert.equal(calls.length, 3);
  assert.equal(new URL(calls[1][0], "http://comfy").searchParams.get("path"), "2_style/a.png");
  assert.equal(new URL(calls[2][0], "http://comfy").searchParams.get("path"), "2_style/a.txt");
  assert.equal(calls[1][1].body, files[0]);
});

test("a failed upload never returns a usable dataset path", async () => {
  const api = { async fetchApi(url) {
    return url.endsWith("/datasets")
      ? new Response(JSON.stringify({ upload_id: "id", path: "/server/partial", extensions: [".png"] }))
      : new Response("Disk full", { status: 507 });
  } };
  await assert.rejects(uploadDataset([{ name: "a.png", webkitRelativePath: "dataset/a.png" }], api), /Disk full/);
});

test("legacy workflow area in pixels migrates to MP while MP stays unchanged", () => {
  const area = { name: "target_megapixels", value: 1 };
  const node = { widgets: [{ name: "WEIGHTS" }, area] };
  migrateTargetArea(node, { widgets_values: ["models", 1048576] });
  assert.equal(area.value, 1.048576);
  area.value = 2;
  migrateTargetArea(node, { widgets_values: ["models", 2] });
  assert.equal(area.value, 2);
});

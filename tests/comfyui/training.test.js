import assert from "node:assert/strict";
import test from "node:test";
import { addDatasetPicker, addResumePicker, migrateTargetArea, serializeTrainingWidgets, uploadDataset } from "../../web/training.js";

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

test("folder button sits directly below dataset_path and opens a directory picker", () => {
  const previous = globalThis.document;
  let clicked = 0;
  const input = { click() { clicked++; }, remove() {} };
  globalThis.document = { createElement: () => input };
  try {
    const node = {
      widgets: [{ name: "dataset_path", value: "/server/data" }, { name: "source", value: "kohya" }],
      addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
      },
    };
    const cleanup = addDatasetPicker(node, {});
    assert.equal(node.widgets[1].type, "button");
    assert.equal(node.widgets[1].serialize, false);
    node.widgets[1].callback();
    assert.equal(clicked, 1);
    assert.equal(input.webkitdirectory, true);
    assert.equal(node.widgets[0].value, "/server/data");
    cleanup();
  } finally {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  }
});

test("the inline folder button never shifts saved workflow parameters", () => {
  const area = { name: "target_megapixels", value: 1 };
  const node = { widgets: [
    { name: "dataset_path", value: "/server/data" },
    { name: "upload", serialize: false, options: { serialize: false } },
    { name: "source", value: "kohya" }, area,
  ] };
  const saved = {};
  serializeTrainingWidgets(node, saved);
  assert.deepEqual(saved.widgets_values, ["/server/data", "kohya", 1]);
  migrateTargetArea(node, { widgets_values: ["/server/data", "kohya", 1048576] });
  assert.equal(area.value, 1.048576);
});

test("old Training values retain their meaning after dataset widgets are removed", () => {
  const old = Array.from({ length: 42 }, (_, i) => i);
  old[7] = 1048576;
  const widgets = old.filter((_value, i) => ![4, 5, 6, 8, 11, 40, 41].includes(i))
    .map((value, i) => ({ name: i === 4 ? "target_megapixels" : `setting_${i}`, value: null }));
  const saved = { widgets_values: old };
  migrateTargetArea({ widgets }, saved);
  assert.equal(widgets[4].value, 1.048576);
  assert.equal(widgets[5].value, 9);
  assert.equal(widgets.at(-1).value, 39);
  assert.equal(saved.widgets_values.length, 35);
});

test("folder workflow drops the removed variant without shifting training settings", () => {
  const old = Array.from({ length: 38 }, (_, i) => i);
  old[5] = 1;
  const widgets = old.filter((_value, i) => ![4, 8, 37].includes(i))
    .map((_value, i) => ({ name: i === 4 ? "target_megapixels" : `setting_${i}` }));
  migrateTargetArea({ widgets }, { widgets_values: old });
  assert.equal(widgets[4].value, 1);
  assert.equal(widgets[5].value, 6);
  assert.equal(widgets.at(-1).value, 36);
});


test("removing cache directory preserves the following cache and training settings", () => {
  const old = Array.from({ length: 37 }, (_, i) => i);
  old[4] = 1;
  const widgets = old.filter((_value, i) => ![7, 36].includes(i))
    .map((_value, i) => ({ name: i === 4 ? "target_megapixels" : `setting_${i}` }));
  migrateTargetArea({ widgets }, { widgets_values: old });
  assert.equal(widgets[6].value, 6);
  assert.equal(widgets[7].value, 8);
  assert.equal(widgets.at(-1).value, 35);
});


test("resume picker uses server directories and only stores the selected path", async () => {
  const previous = globalThis.document;
  const dialogs = [];
  const element = (tag) => ({
    tag, children: [], listeners: {},
    append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; },
    setAttribute() {},
    addEventListener(name, callback) { this.listeners[name] = callback; },
    showModal() {},
    close() { this.listeners.close?.(); },
    remove() {},
  });
  globalThis.document = {
    createElement: element,
    body: { append(item) { dialogs.push(item); } },
  };
  try {
    const resume = { name: "RESUME", value: "" };
    const node = {
      widgets: [resume, { name: "run_token", value: 0 }],
      addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
      },
    };
    const requests = [];
    const cleanup = addResumePicker(node, { async fetchApi(url) {
      requests.push(url);
      return new Response(JSON.stringify({
        path: "/server/output/run/checkpoints/step_00000500",
        parent: "/server/output/run/checkpoints",
        directories: [], checkpoint: true,
      }));
    } });
    assert.equal(node.widgets[1].name, "Browse server checkpoints");
    assert.equal(node.widgets[1].serialize, false);
    await node.widgets[1].callback();
    assert.match(requests[0], /^\/kohakumoss\/checkpoints\?/);
    const choose = dialogs[0].children.find((item) => item.textContent === "Use this checkpoint");
    assert.equal(choose.disabled, false);
    choose.onclick();
    assert.equal(resume.value, "/server/output/run/checkpoints/step_00000500");
    assert.equal(requests.length, 1);
    cleanup();
  } finally {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  }
});


test("hidden advanced overrides are discarded from existing workflows", () => {
  const old = Array.from({ length: 36 }, (_, i) => i);
  old[4] = 1;
  old[35] = '{"LR":99}';
  const widgets = old.slice(0, -1).map((_value, i) => ({
    name: i === 4 ? "target_megapixels" : `setting_${i}`,
  }));
  const saved = { widgets_values: old };
  migrateTargetArea({ widgets }, saved);
  assert.equal(saved.widgets_values.length, 35);
  assert.equal(widgets.at(-1).value, 34);
  assert.ok(!saved.widgets_values.includes('{"LR":99}'));
});

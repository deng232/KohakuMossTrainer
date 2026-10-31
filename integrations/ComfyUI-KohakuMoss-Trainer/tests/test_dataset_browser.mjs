import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

test("native folder picker uploads images and sidecars before committing the server path", async () => {
    const elements = [], uploads = [];
    class Element {
        constructor(tag) { this.tag = tag; this.style = {}; this.textContent = ""; this.events = {}; this.webkitdirectory = false; elements.push(this); }
        append() {}
        addEventListener(name, callback) { this.events[name] = callback; }
        click() { this.clicked = true; }
    }
    class FormData {
        constructor() { this.parts = new Map(); }
        append(name, value) { this.parts.set(name, value); }
        get(name) { return this.parts.get(name); }
    }
    let fail = false;
    const api = { async fetchApi(url, { body }) {
        assert.equal(url, "/kohakumoss/datasets/upload");
        uploads.push([body.get("relative_path"), body.get("upload_id")]);
        return { ok: !fail, status: fail ? 400 : 200, text: async () => "upload failed", json: async () => fail ? { error: "upload failed" } :
            { upload_id: "session", dataset_path: "/input/.kohaku_uploads/session/dataset" } };
    } };
    const context = { api, FormData, document: { createElement: tag => new Element(tag) } };
    const source = readFileSync(new URL("../web/dataset_browser.js", import.meta.url), "utf8")
        .replace(/^import .*;$/gm, "").replace(/^export /gm, "");
    vm.runInNewContext(source, context);
    let changed = 0;
    const node = {
        addDOMWidget(name, type, el, options) {
            return { get value() { return options.getValue(); }, set value(value) { options.setValue(value); } };
        },
        graph: { change() { changed++; } }, setDirtyCanvas() {},
    };
    const { widget } = context.datasetFolderWidget(node, "dataset_path", ["STRING", { default: "/old/dataset" }]);
    const button = elements.find(el => el.tag === "button");
    const input = elements.find(el => el.tag === "input");
    assert.equal(input.type, "file");
    assert.equal(input.webkitdirectory, true);
    assert.equal(input.multiple, true);
    button.onclick();
    assert.equal(input.clicked, true);
    input.files = [
        { name: "photo.png", webkitRelativePath: "dataset/10_person/photo.png" },
        { name: "photo.txt", webkitRelativePath: "dataset/10_person/photo.txt" },
        { name: "model.safetensors", webkitRelativePath: "dataset/model.safetensors" },
    ];
    await input.events.change();
    assert.deepEqual(uploads, [["dataset/10_person/photo.png", ""], ["dataset/10_person/photo.txt", "session"]]);
    assert.equal(widget.value, "/input/.kohaku_uploads/session/dataset");
    assert.equal(changed, 1);
    assert.equal(input.value, "");
    fail = true;
    await input.events.change();
    assert.equal(widget.value, "/input/.kohaku_uploads/session/dataset");
    assert.equal(changed, 1);
    assert.ok(elements.some(el => el.textContent.includes("upload failed")));
});

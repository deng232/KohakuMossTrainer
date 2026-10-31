import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

test("output browser navigates server folders and serializes the selected directory", async () => {
    const elements = [], requests = [];
    class Element {
        constructor(tag) { this.tag = tag; this.style = {}; this.events = {}; elements.push(this); }
        append() {}
        replaceChildren() {}
        setAttribute() {}
        addEventListener(name, fn) { this.events[name] = fn; }
        showModal() {}
        close() { this.events.close?.(); }
        remove() {}
    }
    const context = {
        encodeURIComponent,
        api: { async fetchApi(url) {
            requests.push(url);
            const path = new URL(url, "http://localhost").searchParams.get("path");
            return { ok: true, json: async () => ({ path, parent: path === "/loras" ? null : "/loras", directories: path === "/loras" ? ["/loras/portraits"] : [] }) };
        } },
        document: { createElement: tag => new Element(tag), body: { append() {} } },
    };
    const source = readFileSync(new URL("../web/output_browser.js", import.meta.url), "utf8").replace(/^import .*;$/gm, "").replace(/^export /gm, "");
    vm.runInNewContext(source, context);
    let changes = 0;
    const node = {
        graph: { change() { changes++; } }, setDirtyCanvas() {},
        addDOMWidget(name, type, root, options) {
            return { get value() { return options.getValue(); }, set value(v) { options.setValue(v); } };
        },
    };
    const { widget } = context.outputFolderWidget(node, "output_path", ["STRING", { default: "/loras" }]);
    await elements.find(el => el.textContent === "choose folder").onclick();
    await elements.find(el => el.textContent === "portraits").onclick();
    elements.filter(el => el.textContent === "Select this folder").at(-1).onclick();
    assert.equal(widget.value, "/loras/portraits");
    assert.equal(changes, 1);
    assert.equal(requests.length, 2);
});

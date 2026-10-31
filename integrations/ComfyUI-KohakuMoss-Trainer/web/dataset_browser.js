import { api } from "../../scripts/api.js";

const IMAGE = /\.(jpe?g|png|webp|bmp|jxl|avif)$/i;
const DATASET_FILE = /\.(jpe?g|png|webp|bmp|jxl|avif|txt|caption|json)$/i;

export function datasetFolderWidget(node, name, inputData) {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%";
    const button = document.createElement("button");
    button.textContent = "choose folder";
    button.type = "button";
    button.title = "Upload a dataset folder, including images, captions and subfolders";
    button.style.minHeight = "30px";
    const path = document.createElement("span");
    path.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px";
    const status = document.createElement("span");
    status.style.cssText = "font-size:12px;overflow-wrap:anywhere";
    status.setAttribute?.("role", "status");
    const input = document.createElement("input");
    const supported = "webkitdirectory" in input;
    input.type = "file";
    input.webkitdirectory = true;
    input.multiple = true;
    input.hidden = true;
    root.append(button, path, status, input);
    let value = inputData[1]?.default || "";
    const render = () => {
        path.textContent = value || "No dataset selected";
        path.title = value;
    };
    const widget = node.addDOMWidget(name, "KOHAKU_DATASET_FOLDER", root, {
        getValue: () => value,
        setValue: next => { value = typeof next === "string" ? next : ""; render(); },
        getMinHeight: () => 76,
        getMaxHeight: () => 76,
    });
    button.onclick = () => input.click();
    input.addEventListener("change", async () => {
        const files = Array.from(input.files || []).filter(file => DATASET_FILE.test(file.name));
        if (!files.length) return;
        button.disabled = true;
        try {
            if (!files.some(file => IMAGE.test(file.name))) throw new Error("The selected folder has no supported dataset images.");
            let uploadId = "", datasetPath = "";
            for (const [index, file] of files.entries()) {
                if (!file.webkitRelativePath) throw new Error("This browser did not provide folder-relative file paths.");
                status.textContent = `Uploading ${index + 1}/${files.length}: ${file.webkitRelativePath}`;
                const form = new FormData();
                form.append("upload_id", uploadId);
                form.append("relative_path", file.webkitRelativePath);
                form.append("file", file);
                const response = await api.fetchApi("/kohakumoss/datasets/upload", { method: "POST", body: form });
                if (!response.ok) {
                    const detail = await response.text();
                    throw new Error(`Upload failed (HTTP ${response.status}): ${detail.slice(0, 500)}`);
                }
                const result = await response.json();
                uploadId = result.upload_id;
                datasetPath = result.dataset_path;
            }
            widget.value = datasetPath;
            status.textContent = `Uploaded ${files.length} image/caption files. Ready to queue training.`;
            node.graph?.change();
            node.setDirtyCanvas(true, true);
        } catch (error) {
            status.textContent = error.message;
        } finally {
            button.disabled = false;
            input.value = "";
        }
    });
    if (!supported) {
        button.disabled = true;
        status.textContent = "Folder selection requires a browser supporting webkitdirectory.";
    }
    render();
    return { widget };
}

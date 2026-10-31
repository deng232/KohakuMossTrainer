import { api } from "../../scripts/api.js";

export function outputFolderWidget(node, name, inputData) {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;gap:4px;width:100%";
    const input = document.createElement("input");
    input.type = "text";
    input.value = inputData[1]?.default || "";
    input.style.cssText = "min-width:0;flex:1";
    input.setAttribute("aria-label", "LoRA output directory");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "choose folder";
    root.append(input, button);
    const widget = node.addDOMWidget(name, "KOHAKU_OUTPUT_FOLDER", root, {
        getValue: () => input.value,
        setValue: value => { input.value = typeof value === "string" ? value : ""; },
        getMinHeight: () => 34,
        getMaxHeight: () => 34,
    });
    const changed = () => { node.graph?.change(); node.setDirtyCanvas(true, true); };
    input.addEventListener("change", changed);
    button.onclick = async () => {
        const dialog = document.createElement("dialog");
        dialog.style.cssText = "max-width:85vw;max-height:75vh;overflow:auto";
        dialog.addEventListener("close", () => dialog.remove());
        const content = document.createElement("div");
        const close = document.createElement("button");
        close.textContent = "Close";
        close.onclick = () => dialog.close();
        dialog.append(close, content);
        document.body.append(dialog);
        dialog.showModal();
        async function browse(path) {
            content.replaceChildren();
            try {
                const response = await api.fetchApi(`/kohakumoss/outputs/directories?path=${encodeURIComponent(path)}`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
                const title = document.createElement("p");
                title.textContent = data.path;
                content.append(title);
                function action(text, onclick) {
                    const item = document.createElement("button");
                    item.textContent = text;
                    item.style.cssText = "display:block;margin:8px 0";
                    item.onclick = onclick;
                    content.append(item);
                }
                action("Select this folder", () => { widget.value = data.path; changed(); dialog.close(); });
                if (data.parent) action("..", () => browse(data.parent));
                for (const child of data.directories) action(child.split("/").at(-1), () => browse(child));
            } catch (error) {
                const message = document.createElement("p");
                message.textContent = error.message;
                content.append(message);
                const home = document.createElement("button");
                home.textContent = "Open default LoRA folder";
                home.onclick = () => browse("");
                content.append(home);
            }
        }
        await browse(input.value);
    };
    return { widget };
}

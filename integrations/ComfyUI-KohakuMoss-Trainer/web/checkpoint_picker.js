import { api } from "../../scripts/api.js";

export function checkpointWidget(node, name, inputData) {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;gap:4px;width:100%";
    const input = document.createElement("input");
    input.type = "text";
    input.setAttribute("aria-label", "Complete checkpoint path");
    input.style.cssText = "min-width:0;flex:1";
    input.value = inputData[1]?.default || "latest";
    const button = document.createElement("button");
    button.textContent = "choose checkpoint";
    button.type = "button";
    root.append(input, button);
    const widget = node.addDOMWidget(name, "KOHAKU_CHECKPOINT_FOLDER", root, {
        getValue: () => input.value,
        setValue: value => { input.value = typeof value === "string" ? value : "latest"; },
        getMinHeight: () => 34,
        getMaxHeight: () => 34,
    });
    const changed = jobId => {
        for (const w of node.widgets || []) {
            if (w.name === "start_mode") w.value = "resume_latest";
            if (w.name === "resume_job_id") w.value = jobId;
        }
        node.graph?.change();
        node.setDirtyCanvas(true, true);
    };
    input.addEventListener("change", () => changed(""));
    button.onclick = async () => {
        const dialog = document.createElement("dialog");
        dialog.style.cssText = "max-width:85vw;max-height:75vh;overflow:auto";
        const close = document.createElement("button");
        close.textContent = "Close";
        close.onclick = () => dialog.close();
        dialog.append(close);
        dialog.addEventListener("close", () => dialog.remove());
        document.body.append(dialog);
        dialog.showModal();
        try {
            const response = await api.fetchApi("/kohakumoss/jobs");
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            let count = 0;
            for (const job of data.jobs.filter(job => job.can_resume)) {
                for (const path of job.checkpoints) {
                    const choice = document.createElement("button");
                    choice.style.cssText = "display:block;margin:8px 0;overflow-wrap:anywhere;text-align:left";
                    choice.textContent = `${job.run_name} (${job.status}) · ${path}`;
                    choice.onclick = () => {
                        widget.value = path;
                        changed(job.job_id);
                        dialog.close();
                    };
                    dialog.append(choice);
                    count++;
                }
            }
            if (!count) throw new Error("No complete checkpoints from resumable jobs yet.");
        } catch (error) {
            const message = document.createElement("p");
            message.textContent = error.message;
            dialog.append(message);
        }
    };
    return { widget };
}

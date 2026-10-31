import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { datasetFolderWidget } from "./dataset_browser.js";
import { checkpointWidget } from "./checkpoint_picker.js";
import { outputFolderWidget } from "./output_browser.js";

let disposePanel;

function element(tag, text, parent) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.append(node);
    return node;
}

function plot(canvas, rows) {
    const points = rows.filter(row => Number.isFinite(row.step) && Number.isFinite(row.loss));
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points.length) return;
    const steps = points.map(p => p.step), losses = points.map(p => p.loss);
    const x0 = Math.min(...steps), x1 = Math.max(...steps);
    const y0 = Math.min(...losses), y1 = Math.max(...losses);
    ctx.strokeStyle = "#68c9bc";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, i) => {
        const x = 8 + (point.step - x0) / (x1 - x0 || 1) * (canvas.width - 16);
        const y = canvas.height - 8 - (point.loss - y0) / (y1 - y0 || 1) * (canvas.height - 16);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    canvas.setAttribute("aria-label", `Loss by step, ${points.length} points; latest step ${steps.at(-1)}, loss ${losses.at(-1)}`);
}

function mount(host) {
    disposePanel?.();
    host.replaceChildren();
    const root = element("section", undefined, host);
    root.style.cssText = "padding:12px;display:flex;flex-direction:column;gap:10px;overflow:auto;height:100%;box-sizing:border-box";
    element("h3", "KohakuMoss Trainer", root);
    element("p", "Start with Queue Prompt. Changes to the workflow apply only to the next new job.", root);
    element("p", "Training shares GPU memory with inference. Use this UI only with trusted users; do not expose an unauthenticated ComfyUI server to the internet.", root);
    const label = element("label", "Job history ", root);
    const select = element("select", undefined, label);
    select.setAttribute("aria-label", "Select training job");
    select.style.maxWidth = "100%";
    const status = element("p", "No jobs yet.", root);
    const actions = element("div", undefined, root);
    const stop = element("button", "Stop", actions);
    const resume = element("button", "Continue training", actions);
    const checkpoint = element("select", undefined, root);
    checkpoint.setAttribute("aria-label", "Complete checkpoint to resume");
    const stepsLabel = element("label", "Additional steps ", root);
    const additional = element("input", undefined, stepsLabel);
    additional.type = "number";
    additional.min = "1";
    additional.step = "1";
    additional.value = "1000";
    element("p", "Stop uses SIGTERM, then forces the owned process group to exit after 30 seconds if needed. It does not guarantee a new checkpoint. Continue uses the selected checkpoint and additional steps, keeping the original job settings and LoRA output path.", root);
    const error = element("pre", "", root);
    error.setAttribute("role", "status");
    const metrics = element("pre", "", root);
    const chart = element("canvas", undefined, root);
    chart.width = 480;
    chart.height = 140;
    chart.style.cssText = "width:100%;background:#202328";
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", "Loss by step; no metrics yet");
    const paths = element("pre", "", root);
    const logs = element("pre", "", root);
    logs.setAttribute("aria-label", "Recent worker logs");
    logs.style.cssText = "max-height:360px;overflow:auto";
    for (const pre of [error, metrics, paths, logs]) {
        pre.style.whiteSpace = "pre-wrap";
        pre.style.overflowWrap = "anywhere";
    }

    let selected = "", offset = 0, generation = 0, selectionVersion = 0, timer, controller;
    let disposed = false, mounted = false, controlling = false;
    const request = async (url, options = {}, signal = controller?.signal) => {
        const response = await api.fetchApi(`/kohakumoss/jobs${url}`, { ...options, signal });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        return data;
    };
    const dispose = () => {
        disposed = true;
        clearTimeout(timer);
        controller?.abort();
    };
    disposePanel = dispose;

    async function poll() {
        if (disposed) return;
        if (root.isConnected) mounted = true;
        else if (mounted) return dispose();
        controller = new AbortController();
        try {
            const { jobs } = await request("");
            if (disposed) return;
            const previous = selected;
            selected = jobs.some(job => job.job_id === selected) ? selected : jobs[0]?.job_id || "";
            select.replaceChildren(...jobs.map(job => {
                const option = element("option", `${job.run_name} — ${job.status}`);
                option.value = job.job_id;
                return option;
            }));
            select.value = selected;
            if (selected !== previous) { offset = 0; generation = 0; logs.textContent = ""; }
            const job = jobs.find(job => job.job_id === selected);
            if (!job) {
                stop.disabled = resume.disabled = true;
                return;
            }
            const chosen = selected === previous ? checkpoint.value : "";
            checkpoint.replaceChildren(...job.checkpoints.map(path => {
                const option = element("option", path);
                option.value = path;
                return option;
            }));
            checkpoint.value = job.checkpoints.includes(chosen) ? chosen : job.checkpoints.at(-1) || "";
            status.textContent = `${job.status} · ${job.job_id}`;
            stop.disabled = controlling || ["completed", "failed", "stopped", "interrupted", "unknown", "stopping"].includes(job.status);
            resume.disabled = controlling || !job.can_resume;
            error.textContent = [job.error, job.stop_note, ...(job.warnings || [])].filter(Boolean).join("\n");
            metrics.textContent = `Step ${job.metrics.step ?? "—"} / ${job.settings.steps}` +
                (job.resume_step ? ` · resumed from ${job.resume_step}` : "") + "\n" +
                Object.entries(job.metrics).filter(([key]) => !["step", "time"].includes(key))
                    .map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`).join("\n");
            paths.textContent = [`Run: ${job.run_dir}`, `Published: ${job.published || "—"}`,
                `Exports:\n${job.exports.join("\n") || "—"}`,
                `Complete checkpoints:\n${job.checkpoints.join("\n") || "—"}`].join("\n");
            const jobId = selected, version = selectionVersion;
            const [tail, history] = await Promise.all([
                request(`/${jobId}/logs?offset=${offset}&generation=${generation}&limit=200`),
                request(`/${jobId}/metrics?limit=1000`),
            ]);
            if (disposed || selected !== jobId || selectionVersion !== version) return;
            if (generation !== tail.generation) logs.textContent = "";
            generation = tail.generation;
            offset = tail.offset;
            logs.textContent = (logs.textContent + tail.text).slice(-48000);
            plot(chart, history.rows);
        } catch (exception) {
            if (exception.name !== "AbortError") error.textContent = exception.message;
        } finally {
            if (!disposed) timer = setTimeout(poll, 2000);
        }
    }
    // A single poller belongs to the sidebar, independent of workflow node count.
    select.onchange = () => {
        selected = select.value;
        selectionVersion += 1;
        offset = generation = 0;
        logs.textContent = "";
    };
    for (const [button, action] of [[stop, "stop"], [resume, "resume"]]) {
        button.onclick = async () => {
            if (controlling || !selected) return;
            controlling = true;
            stop.disabled = resume.disabled = true;
            try {
                const options = { method: "POST" };
                if (action === "resume") {
                    options.headers = { "Content-Type": "application/json" };
                    options.body = JSON.stringify({ checkpoint: checkpoint.value, additional_steps: Number(additional.value) });
                }
                await request(`/${selected}/${action}`, options, null);
            }
            catch (exception) { error.textContent = exception.message; }
            finally { controlling = false; }
        };
    }
    void poll();
}

app.registerExtension({
    name: "KohakuMoss.Trainer",
    getCustomWidgets() {
        return { KOHAKU_DATASET_FOLDER: datasetFolderWidget, KOHAKU_CHECKPOINT_FOLDER: checkpointWidget, KOHAKU_OUTPUT_FOLDER: outputFolderWidget };
    },
    nodeCreated(node) {
        if (node.comfyClass !== "KohakuMossTrainer") return;
        const learningRate = node.widgets.find(w => w.name === "learning_rate");
        // The frontend can override schema rounding with its global float settings.
        if (learningRate) Object.assign(learningRate.options, { precision: 12, round: false });
        const mode = node.widgets.find(w => w.name === "start_mode");
        const checkpoint = node.widgets.find(w => w.name === "resume_checkpoint");
        if (!mode || !checkpoint) return;
        const computeSize = checkpoint.computeSize;
        const update = () => {
            const hidden = mode.value === "new";
            checkpoint.hidden = hidden;
            if (checkpoint.element) checkpoint.element.hidden = hidden;
            checkpoint.computeSize = hidden ? () => [0, -4] : computeSize;
            node.setDirtyCanvas(true, true);
        };
        const callback = mode.callback;
        mode.callback = function (...args) { callback?.apply(this, args); update(); };
        const configure = node.onConfigure;
        node.onConfigure = function (...args) { configure?.apply(this, args); update(); };
        update();
    },
    setup() {
        app.extensionManager.registerSidebarTab({
            id: "kohakumoss-trainer", icon: "pi pi-chart-line", title: "KohakuMoss",
            tooltip: "KohakuMoss training jobs", type: "custom", render: mount,
        });
    },
});

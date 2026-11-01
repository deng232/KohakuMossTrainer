import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import uPlot from "./vendor/uPlot.esm.js";
import { MonitorState, upstream, workflowKey } from "./monitor.js";

const views = new Set();
const style = document.createElement("link");
style.rel = "stylesheet";
style.href = new URL("./monitor.css", import.meta.url).href;
document.head.append(style);

function workflowId(graph = app.graph) {
  return workflowKey(graph);
}

function chain(node, name, callback) {
  const prior = node[name];
  node[name] = function (...args) {
    const result = prior?.apply(this, args);
    callback(...args);
    return result;
  };
}

class MonitorView {
  constructor(node) {
    this.node = node;
    this.generation = 0;
    this.removed = false;
    this.timer = null;
    this.lastPaint = 0;
    this.plot = null;
    this.seriesKey = "";
    this.listeners = [];
    this.element = document.createElement("div");
    this.element.className = "kohakumoss-monitor";
    this.status = document.createElement("div");
    this.status.setAttribute("role", "status");
    this.status.setAttribute("aria-live", "polite");
    this.chart = document.createElement("div");
    this.chart.className = "kohakumoss-chart";
    this.element.append(this.status, this.chart);
    this.widget = node.addDOMWidget("kohakumoss_monitor", "kohakumoss_monitor", this.element, {
      serialize: false, getMinHeight: () => 340, hideOnZoom: false,
    });
    node.setSize([Math.max(node.size[0], 620), Math.max(node.size[1], 500)]);
    this.observe = new ResizeObserver(() => this.paintSoon());
    this.observe.observe(this.element);
    for (const name of ["kohakumoss.job_started", "kohakumoss.metrics", "kohakumoss.status"]) {
      this.listen(name, (event) => {
        this.bind();
        if (this.state?.receive(event.detail)) this.paintSoon();
      });
    }
    this.listen("reconnected", () => this.restore());
    chain(node, "onConnectionsChange", () => queueMicrotask(() => this.bind()));
    chain(node, "onWidgetChanged", () => this.paintSoon());
    chain(node, "onRemoved", () => this.destroy());
    views.add(this);
    queueMicrotask(() => this.bind());
    this.paintSoon();
  }

  listen(name, callback) {
    const add = api.addCustomEventListener ?? api.addEventListener;
    add.call(api, name, callback);
    this.listeners.push([name, callback]);
  }

  bind() {
    if (this.removed || !this.node.graph) return;
    const origin = upstream(this.node);
    const workflow = workflowId(this.node.graph);
    if (origin === this.state?.origin && workflow === this.state?.workflow) return;
    this.generation++;
    this.state = new MonitorState(origin, workflow);
    this.paintSoon();
    if (origin !== null) this.restore();
  }

  async restore() {
    if (this.removed || this.state?.origin == null) return;
    const generation = ++this.generation;
    const state = this.state;
    const query = new URLSearchParams({ origin_node_id: state.origin, workflow_id: state.workflow });
    try {
      const list = await api.fetchApi(`/kohakumoss/jobs?${query}`);
      if (!list.ok) throw new Error(`Job recovery failed (${list.status})`);
      const { jobs } = await list.json();
      if (this.removed || generation !== this.generation || !jobs.length) return;
      const history = await api.fetchApi(`/kohakumoss/jobs/${encodeURIComponent(jobs[0].job_id)}/history`);
      if (!history.ok) throw new Error(`History recovery failed (${history.status})`);
      const data = await history.json();
      if (!this.removed && generation === this.generation && state.hydrate(data)) this.paintSoon();
    } catch (error) {
      if (!this.removed && generation === this.generation) {
        this.status.textContent = error.message;
      }
    }
  }

  setting(name, fallback) {
    return this.node.widgets?.find((widget) => widget.name === name)?.value ?? fallback;
  }

  paintSoon() {
    if (this.removed || this.timer !== null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.lastPaint = performance.now();
      this.paint();
    }, Math.max(0, 500 - (performance.now() - this.lastPaint)));
  }

  paint() {
    const snap = this.state?.snapshot;
    if (!snap) {
      this.status.textContent = "Waiting for training";
      this.plot?.destroy();
      this.plot = null;
      return;
    }
    const numeric = (value) => typeof value === "number" && Number.isFinite(value) ? value.toPrecision(5) : "--";
    const step = snap.last_metric_step ?? "--";
    const phase = snap.phase?.startsWith("CACHE_") ? ` ${snap.phase}: ${snap.cache_completed ?? 0} written` : "";
    this.status.textContent = `${snap.status}${phase} | step ${step}/${snap.total_steps ?? "--"} | loss ${numeric(snap.last_metrics?.loss)} | lr ${numeric(snap.last_metrics?.lr)}${snap.error ? ` | ${snap.error}` : ""}`;
    this.state.limit = Math.min(10_000, Math.max(10, Number(this.setting("history_points", 10_000))));
    this.state.trim();
    if (this.state.truncated) this.status.textContent += " | showing retained history";
    const { keys, data } = this.state.columns({
      showLR: this.setting("show_lr", true), showVRAM: this.setting("show_vram", true),
      smoothing: this.setting("smoothing", 1),
    });
    const width = Math.max(250, Math.floor(this.element.clientWidth || 560));
    const height = 270;
    if (!this.plot || this.seriesKey !== keys.join()) {
      this.plot?.destroy();
      const colors = ["#5bc0eb", "#ffb454", "#b292ff"];
      this.plot = new uPlot({
        width, height,
        scales: { x: { time: false }, loss: { auto: true }, lr: { auto: true }, vram: { auto: true } },
        axes: [{ label: "Step", stroke: "#bbb" }, { scale: "loss", stroke: colors[0] }],
        series: [{ label: "Step" }, ...keys.map((key, i) => ({
          label: key, stroke: colors[i], scale: key.startsWith("vram") ? "vram" : key,
          value: (_plot, value) => numeric(value),
        }))],
      }, data, this.chart);
      this.seriesKey = keys.join();
    } else {
      this.plot.setSize({ width, height });
      this.plot.setData(data);
    }
  }

  destroy() {
    if (this.removed) return;
    this.removed = true;
    this.generation++;
    clearTimeout(this.timer);
    this.observe.disconnect();
    for (const [name, callback] of this.listeners) api.removeEventListener(name, callback);
    this.listeners.length = 0;
    this.plot?.destroy();
    this.element.remove();
    views.delete(this);
  }
}

app.registerExtension({
  name: "KohakuMoss.LiveMonitor",
  beforeConfigureGraph() {
    for (const view of [...views]) view.destroy();
  },
  nodeCreated(node) {
    if (node.comfyClass === "KohakuMossMonitor" || node.type === "KohakuMossMonitor") new MonitorView(node);
  },
  afterConfigureGraph() {
    workflowId();
    for (const view of views) view.bind();
  },
});

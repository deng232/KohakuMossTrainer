export function upstream(node) {
  const input = node.inputs?.find((item) => item.name === "training_job") ?? node.inputs?.[0];
  if (input?.link == null || !node.graph) return null;
  const link = node.graph.links?.[input.link];
  const origin = link?.origin_id;
  if (origin == null || node.graph.getNodeById(origin)?.type !== "KohakuMossTraining") return null;
  return String(origin);
}

export function workflowKey(graph, cryptoAPI = globalThis.crypto) {
  graph.extra ??= {};
  return graph.extra.kohakumoss_id ??= cryptoAPI.randomUUID?.() ??
    Array.from(cryptoAPI.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

const finite = (value) => typeof value === "number" && Number.isFinite(value) ? value : null;

export class MonitorState {
  constructor(origin, workflow, limit = 10_000) {
    this.origin = origin;
    this.workflow = workflow;
    this.limit = limit;
    this.snapshot = null;
    this.rows = new Map();
    this.truncated = false;
  }

  matches(event) {
    return String(event.origin_node_id) === this.origin && event.workflow_id === this.workflow;
  }

  adopt(event) {
    if (!this.matches(event)) return false;
    if (this.snapshot?.job_id !== event.job_id) {
      if (this.snapshot && event.started_at <= this.snapshot.started_at) return false;
      this.rows.clear();
      this.truncated = false;
      this.snapshot = null;
    }
    return true;
  }

  receive(event) {
    if (!this.adopt(event) || event.event_seq <= (this.snapshot?.event_seq ?? 0)) return false;
    this.snapshot = event;
    if (event.last_metric_step != null && Object.keys(event.last_metrics ?? {}).length) {
      this.rows.set(event.last_metric_step, {
        step: event.last_metric_step, timestamp: event.timestamp,
        metrics: { ...event.last_metrics }, event_seq: event.event_seq,
      });
    }
    this.trim();
    return true;
  }

  hydrate(history) {
    if (!this.adopt(history.snapshot)) return false;
    const newer = [...this.rows.values()].filter((row) => row.event_seq > history.event_seq);
    this.rows = new Map(history.rows.map((row) => [row.step, row]));
    for (const row of newer) this.rows.set(row.step, row);
    if (!this.snapshot || history.snapshot.event_seq >= this.snapshot.event_seq) this.snapshot = history.snapshot;
    this.truncated = Boolean(history.truncated);
    this.trim();
    return true;
  }

  trim() {
    const steps = [...this.rows.keys()].sort((a, b) => a - b);
    while (steps.length > this.limit) {
      this.rows.delete(steps.shift());
      this.truncated = true;
    }
  }

  columns({ showLR = true, showVRAM = true, smoothing = 1 } = {}) {
    const rows = [...this.rows.values()].sort((a, b) => a.step - b.step);
    const keys = ["loss"];
    if (showLR) keys.push("lr");
    if (showVRAM) {
      const vram = ["vram_peak_alloc", "vram_allocated", "vram_peak_reserved", "vram_reserved"].find((key) => rows.some((r) => finite(r.metrics[key]) !== null));
      if (vram) keys.push(vram);
    }
    const data = [rows.map((r) => r.step), ...keys.map((key) => rows.map((r) => finite(r.metrics[key])))];
    const window = Math.max(1, Math.floor(smoothing));
    if (window > 1) {
      const raw = data[1];
      let sum = 0, count = 0;
      data[1] = raw.map((value, i) => {
        if (value !== null) { sum += value; count++; }
        if (i >= window && raw[i - window] !== null) { sum -= raw[i - window]; count--; }
        return value === null ? null : sum / count;
      });
    }
    return { keys, data };
  }
}

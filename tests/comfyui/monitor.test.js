import assert from "node:assert/strict";
import test from "node:test";
import { MonitorState, upstream, workflowKey } from "../../web/monitor.js";

const event = (seq, step, extra = {}) => ({
  job_id: "job-a", origin_node_id: "17", workflow_id: "workflow", event_seq: seq,
  started_at: 1, status: "TRAINING", last_metric_step: step,
  last_metrics: { loss: step / 10, lr: 0.001 }, ...extra,
});

test("routes without a backend monitor output, deduplicates and rejects other jobs", () => {
  const state = new MonitorState("17", "workflow");
  assert.equal(state.receive(event(1, 5)), true);
  assert.equal(state.receive(event(1, 5)), false);
  assert.equal(state.receive(event(2, 10, { origin_node_id: "18" })), false);
  assert.equal(state.receive(event(2, 10, { workflow_id: "elsewhere" })), false);
  assert.deepEqual(state.columns().data[0], [5]);
  assert.equal(state.receive(event(1, 1, { job_id: "job-b", started_at: 2 })), true);
  assert.equal(state.receive(event(3, 15)), false);
  assert.deepEqual(state.columns().data[0], [1]);
});

test("history hydration keeps newer live points and ignores stale status", () => {
  const state = new MonitorState("17", "workflow");
  state.receive(event(4, 15));
  state.hydrate({
    job_id: "job-a", event_seq: 3, snapshot: event(3, 10),
    rows: [{ step: 5, metrics: { loss: 0.5 } }, { step: 10, metrics: { loss: 1 } }],
  });
  assert.equal(state.snapshot.event_seq, 4);
  const { data } = state.columns({ showLR: true, showVRAM: true });
  assert.deepEqual(data[0], [5, 10, 15]);
  assert.deepEqual(data[2], [null, null, 0.001]);
  assert.ok(data.every((column) => column.length === 3));
});

test("bounds history and smooths only real values without altering raw loss", () => {
  const state = new MonitorState("17", "workflow", 2);
  state.receive(event(1, 1));
  state.receive(event(2, 2, { last_metrics: { loss: null } }));
  state.receive(event(3, 3));
  assert.deepEqual(state.columns({ smoothing: 2 }).data[1], [null, 0.3]);
  assert.deepEqual(state.columns().data[0], [2, 3]);
  assert.equal(state.rows.get(3).metrics.loss, 0.3);
});

test("uses the real upstream link and clears on disconnection", () => {
  const graph = { links: { 7: { origin_id: 17 } }, getNodeById: () => ({ type: "KohakuMossTraining" }) };
  assert.equal(upstream({ inputs: [{ name: "training_job", link: 7 }], graph }), "17");
  assert.equal(upstream({ inputs: [{ link: null }], graph }), null);
});

test("bench-level native VRAM fields are plotted", () => {
  const state = new MonitorState("17", "workflow");
  state.receive(event(1, 1, { last_metrics: { loss: 0.42, vram_peak_alloc: 3.5 } }));
  assert.deepEqual(state.columns({ showLR: false }).keys, ["loss", "vram_peak_alloc"]);
  assert.deepEqual(state.columns({ showLR: false }).data[2], [3.5]);
});

test("workflow identity works on LAN HTTP without randomUUID and stays serialized", () => {
  const graph = {};
  const insecureCrypto = { getRandomValues: (bytes) => bytes.fill(7) };
  assert.equal(workflowKey(graph, insecureCrypto), "07070707070707070707070707070707");
  assert.equal(workflowKey(graph, {}), graph.extra.kohakumoss_id);
});

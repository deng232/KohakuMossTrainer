export async function uploadDataset(files, api, progress = () => {}, signal) {
  const created = await api.fetchApi("/kohakumoss/datasets", { method: "POST", signal });
  if (!created.ok) throw new Error(await created.text());
  const dataset = await created.json();
  const allowed = new Set(dataset.extensions);
  const selected = files.filter((file) => allowed.has(file.name.slice(file.name.lastIndexOf(".")).toLowerCase()));
  if (!selected.length) throw new Error("The selected folder has no supported images or captions.");
  for (let i = 0; i < selected.length; i++) {
    const file = selected[i];
    const path = file.webkitRelativePath?.split("/").slice(1).join("/") || file.name;
    const query = new URLSearchParams({ path });
    const reply = await api.fetchApi(`/kohakumoss/datasets/${dataset.upload_id}/files?${query}`, {
      method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" }, signal,
    });
    if (!reply.ok) throw new Error(`Upload failed: ${path}: ${await reply.text()}`);
    progress(i + 1, selected.length);
  }
  return dataset.path;
}

export function migrateTargetArea(node, saved) {
  const widgets = node.widgets?.filter((widget) => widget.serialize !== false && widget.options?.serialize !== false) ?? [];
  // Old Training nodes included dataset_path, source, CAPTION and DATASETS widgets.
  if (saved?.widgets_values?.length === 42 && widgets.length === 35) {
    saved.widgets_values = saved.widgets_values.filter((_value, i) => ![4, 5, 6, 8, 41].includes(i));
  }
  if (saved?.widgets_values?.length === 38 && widgets.length === 35) {
    saved.widgets_values = saved.widgets_values.filter((_value, i) => i !== 4);
  }
  if (saved?.widgets_values?.length === 37 && widgets.length === 35) {
    saved.widgets_values = saved.widgets_values.filter((_value, i) => i !== 7);
  }
  if (saved?.widgets_values?.length === 36 && widgets.length === 35) {
    saved.widgets_values = saved.widgets_values.slice(0, -1);
    widgets.forEach((widget, i) => { widget.value = saved.widgets_values[i]; });
  }
  const index = widgets.findIndex((widget) => widget.name === "target_megapixels");
  const value = saved?.widgets_values?.[index];
  if (index >= 0 && typeof value === "number" && value > 64) widgets[index].value = value / 1_000_000;
}

export function serializeTrainingWidgets(node, saved) {
  saved.widgets_values = node.widgets.filter((widget) => widget.serialize !== false && widget.options?.serialize !== false).map((widget) => widget.value);
}

export function addDatasetPicker(node, api) {
  const input = document.createElement("input");
  input.type = "file";
  input.webkitdirectory = true;
  input.multiple = true;
  let controller = null;
  let removed = false;
  const button = node.addWidget("button", "Choose dataset folder (upload)", null, () => {
    if (!controller) input.click();
  }, { serialize: false });
  button.serialize = false;
  const pathIndex = node.widgets.findIndex((widget) => widget.name === "dataset_path");
  if (pathIndex >= 0) {
    node.widgets.splice(node.widgets.indexOf(button), 1);
    node.widgets.splice(pathIndex + 1, 0, button);
  }
  input.onchange = async () => {
    const files = Array.from(input.files ?? []);
    if (!files.length || controller || removed) return;
    controller = new AbortController();
    button.name = "Uploading dataset...";
    try {
      const path = await uploadDataset(files, api, (done, total) => {
        button.name = `Uploading dataset ${done}/${total}`;
        node.graph?.setDirtyCanvas(true, true);
      }, controller.signal);
      if (!removed) {
        const widget = node.widgets.find((item) => item.name === "dataset_path");
        widget.value = path;
        widget.callback?.(path);
      }
    } catch (error) {
      if (!removed && error.name !== "AbortError") window.alert(error.message);
    } finally {
      controller = null;
      input.value = "";
      button.name = "Choose dataset folder (upload)";
      node.graph?.setDirtyCanvas(true, true);
    }
  };
  return () => {
    removed = true;
    controller?.abort();
    input.onchange = null;
    input.remove();
  };
}


export function addResumePicker(node, api) {
  let dialog = null;
  let controller = null;
  let removed = false;
  const button = node.addWidget("button", "Browse server checkpoints", null, async () => {
    if (dialog || removed) return;
    dialog = document.createElement("dialog");
    dialog.className = "kohakumoss-checkpoints";
    const title = document.createElement("h3");
    title.textContent = "Server checkpoint folder";
    const path = document.createElement("div");
    const status = document.createElement("div");
    status.setAttribute("role", "status");
    const list = document.createElement("div");
    const up = document.createElement("button");
    up.textContent = "Parent folder";
    const select = document.createElement("button");
    select.textContent = "Use this checkpoint";
    select.disabled = true;
    const close = document.createElement("button");
    close.textContent = "Cancel";
    let current = null;
    async function browse(directory = "") {
      controller?.abort();
      controller = new AbortController();
      const active = controller;
      select.disabled = true;
      up.disabled = true;
      list.replaceChildren();
      status.textContent = "Loading server folders...";
      try {
        const reply = await api.fetchApi(`/kohakumoss/checkpoints?${new URLSearchParams({ path: directory })}`, { signal: active.signal });
        if (!reply.ok) throw new Error(await reply.text());
        const data = await reply.json();
        if (active.signal.aborted || !dialog) return;
        current = data;
        path.textContent = data.path;
        select.disabled = !data.checkpoint;
        up.disabled = !data.parent;
        status.textContent = data.checkpoint ? "Checkpoint found" : "Open a run's checkpoints/step_XXXXXXXX folder.";
        for (const name of data.directories) {
          const entry = document.createElement("button");
          entry.textContent = name;
          entry.onclick = () => browse(`${data.path}/${name}`);
          list.append(entry);
        }
      } catch (error) {
        if (!active.signal.aborted && dialog) status.textContent = error.message;
      }
    }
    up.onclick = () => { if (current?.parent) browse(current.parent); };
    select.onclick = () => {
      if (!current?.checkpoint) return;
      const widget = node.widgets.find((item) => item.name === "RESUME");
      widget.value = current.path;
      widget.callback?.(current.path);
      node.graph?.setDirtyCanvas(true, true);
      dialog.close();
    };
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => {
      controller?.abort();
      dialog?.remove();
      dialog = null;
    }, { once: true });
    dialog.append(title, path, status, up, list, select, close);
    document.body.append(dialog);
    dialog.showModal();
    await browse();
  }, { serialize: false });
  button.serialize = false;
  const index = node.widgets.findIndex((widget) => widget.name === "RESUME");
  node.widgets.splice(node.widgets.indexOf(button), 1);
  node.widgets.splice(index + 1, 0, button);
  return () => {
    removed = true;
    controller?.abort();
    dialog?.close();
    dialog?.remove();
    dialog = null;
  };
}

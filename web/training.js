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
  const index = node.widgets?.findIndex((widget) => widget.name === "target_megapixels");
  const value = saved?.widgets_values?.[index];
  if (index >= 0 && typeof value === "number" && value > 64) node.widgets[index].value = value / 1_000_000;
}

export function addDatasetPicker(node, api) {
  const input = document.createElement("input");
  input.type = "file";
  input.webkitdirectory = true;
  input.multiple = true;
  let controller = null;
  let removed = false;
  const button = node.addWidget("button", "Upload dataset folder", null, () => {
    if (!controller) input.click();
  }, { serialize: false });
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
      button.name = "Upload dataset folder";
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

const createDialog = document.getElementById("createProjectDialog");
const openCreateDialog = document.getElementById("openCreateDialog");
const editDialog = document.getElementById("editProjectDialog");
const editForm = document.getElementById("editProjectForm");
const classesDialog = document.getElementById("classesDialog");
const classesForm = document.getElementById("classesForm");
const editClassesButton = document.getElementById("editClassesButton");

if (createDialog && openCreateDialog) {
  openCreateDialog.addEventListener("click", () => createDialog.showModal());
  createDialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => createDialog.close());
  });
}

if (editDialog && editForm) {
  editDialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => editDialog.close());
  });

  document.querySelectorAll("[data-edit-project]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      editForm.action = `/project/${button.dataset.directory}/edit`;
      editForm.elements.name.value = button.dataset.name || "";
      editForm.elements.description.value = button.dataset.description || "";
      document.querySelectorAll(".menu-panel").forEach((panel) => {
        panel.hidden = true;
      });
      editDialog.showModal();
    });
  });
}

if (classesDialog && classesForm && editClassesButton) {
  editClassesButton.addEventListener("click", () => classesDialog.showModal());
  classesDialog.querySelectorAll("[data-close-classes]").forEach((button) => {
    button.addEventListener("click", () => classesDialog.close());
  });
  classesForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const project = document.querySelector("[data-project]")?.dataset.project;
    const content = classesForm.elements.content.value;
    const response = await fetch(`/project/${project}/classes`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({content}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      alert(data.error || "保存失败");
      return;
    }
    const status = document.querySelector("[data-classes-status]");
    if (status) status.textContent = "已上传";
    const buttonLabel = document.querySelector("[data-classes-edit-label]");
    if (buttonLabel) buttonLabel.textContent = "编辑";
    classesDialog.close();
  });
}

document.querySelectorAll("[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!confirm(form.dataset.confirm || "确认执行该操作？")) {
      event.preventDefault();
    }
  });
});

document.querySelectorAll(".menu-button").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    const panel = button.parentElement.querySelector(".menu-panel");
    const isHidden = panel.hidden;
    document.querySelectorAll(".menu-panel").forEach((item) => {
      item.hidden = true;
    });
    document.querySelectorAll(".menu-button").forEach((item) => {
      item.setAttribute("aria-expanded", "false");
    });
    panel.hidden = !isHidden;
    button.setAttribute("aria-expanded", String(isHidden));
  });
});

document.addEventListener("click", () => {
  document.querySelectorAll(".menu-panel").forEach((panel) => {
    panel.hidden = true;
  });
  document.querySelectorAll(".menu-button").forEach((button) => {
    button.setAttribute("aria-expanded", "false");
  });
});

function setActionEnabled(selector, enabled) {
  const link = document.querySelector(selector);
  if (link) {
    link.classList.toggle("disabled", !enabled);
  }
}

function setAnnotateReady({imagesReady} = {}) {
  const link = document.querySelector("[data-image-action]");
  if (!link) {
    return;
  }
  if (typeof imagesReady === "boolean") {
    link.dataset.imagesReady = imagesReady ? "1" : "0";
  }
  const ready = link.dataset.imagesReady === "1";
  link.classList.toggle("disabled", !ready);
  link.title = ready ? "进入标注" : "请先上传图片";
}

function fileEntryFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

function readDirectoryEntries(reader) {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

async function filesFromEntry(entry, prefix = "") {
  if (entry.isFile) {
    const file = await fileEntryFile(entry);
    return [{file, path: `${prefix}${file.name}`}];
  }

  if (!entry.isDirectory) {
    return [];
  }

  const reader = entry.createReader();
  const files = [];
  while (true) {
    const entries = await readDirectoryEntries(reader);
    if (!entries.length) {
      break;
    }
    for (const child of entries) {
      files.push(...await filesFromEntry(child, `${prefix}${entry.name}/`));
    }
  }
  return files;
}

function filesFromFileList(files) {
  return Array.from(files).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));
}

async function filesFromDataTransfer(dataTransfer) {
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);

  if (!entries.length) {
    return filesFromFileList(dataTransfer.files);
  }

  const files = [];
  for (const entry of entries) {
    files.push(...await filesFromEntry(entry));
  }
  return files;
}

async function uploadFiles(zone, files) {
  const uploads = Array.from(files);
  if (!uploads.length) {
    return;
  }

  const project = document.querySelector("[data-project]")?.dataset.project;
  const kind = zone.dataset.uploadKind;
  if (!project || !kind) {
    return;
  }

  const formData = new FormData();
  uploads.forEach((item) => {
    const file = item.file || item;
    const path = item.path || file.webkitRelativePath || file.name;
    formData.append("files", file, path);
  });
  zone.classList.add("uploading");

  try {
    const response = await fetch(`/project/${project}/upload/${kind}`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "上传失败");
    }
    if (kind === "images") {
      document.querySelector("[data-image-count]").textContent = `${data.count} 个文件`;
      setAnnotateReady({imagesReady: data.count > 0});
    } else if (kind === "model") {
      document.querySelector("[data-model-count]").textContent = `${data.count} 个文件`;
      setActionEnabled("[data-model-action]", data.count > 0);
    } else if (kind === "classes") {
      const label = zone.querySelector("[data-upload-label]");
      if (label) {
        label.textContent = "classes.txt 已上传";
      }
      const status = document.querySelector("[data-classes-status]");
      if (status) status.textContent = "已上传";
      const buttonLabel = document.querySelector("[data-classes-edit-label]");
      if (buttonLabel) buttonLabel.textContent = "编辑";
    }
  } catch (error) {
    alert(error.message);
  } finally {
    zone.classList.remove("uploading");
  }
}

document.querySelectorAll("[data-upload-zone]").forEach((zone) => {
  const input = zone.querySelector("input");
  const panel = zone.closest(".upload-panel");
  const directoryInput = panel?.querySelector("[data-directory-input]");
  const directoryButton = panel?.querySelector("[data-directory-button]");

  input.addEventListener("change", () => uploadFiles(zone, filesFromFileList(input.files)));
  directoryInput?.addEventListener("change", () => uploadFiles(zone, filesFromFileList(directoryInput.files)));
  directoryButton?.addEventListener("click", () => directoryInput?.click());
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    uploadFiles(zone, await filesFromDataTransfer(event.dataTransfer));
  });
});

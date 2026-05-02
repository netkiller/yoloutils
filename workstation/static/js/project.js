const createDialog = document.getElementById("createProjectDialog");
const openCreateDialog = document.getElementById("openCreateDialog");

if (createDialog && openCreateDialog) {
  openCreateDialog.addEventListener("click", () => createDialog.showModal());
  createDialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => createDialog.close());
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

async function uploadFiles(zone, files) {
  if (!files.length) {
    return;
  }

  const project = document.querySelector("[data-project]")?.dataset.project;
  const kind = zone.dataset.uploadKind;
  if (!project || !kind) {
    return;
  }

  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
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
      setActionEnabled("[data-image-action]", data.count > 0);
    } else {
      document.querySelector("[data-model-count]").textContent = `${data.count} 个文件`;
      setActionEnabled("[data-model-action]", data.count > 0);
    }
  } catch (error) {
    alert(error.message);
  } finally {
    zone.classList.remove("uploading");
  }
}

document.querySelectorAll("[data-upload-zone]").forEach((zone) => {
  const input = zone.querySelector("input");
  input.addEventListener("change", () => uploadFiles(zone, input.files));
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragging");
    uploadFiles(zone, event.dataTransfer.files);
  });
});

const createDialog = document.getElementById("createProjectDialog");
const openCreateDialog = document.getElementById("openCreateDialog");
const editDialog = document.getElementById("editProjectDialog");
const editForm = document.getElementById("editProjectForm");
const classesDialog = document.getElementById("classesDialog");
const classesForm = document.getElementById("classesForm");
const editClassesButton = document.getElementById("editClassesButton");
const sftpPanel = document.querySelector("[data-sftp-path]");

function copyFeedback(button, text) {
  if (!button || !text) {
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    button.textContent = "✓";
    setTimeout(() => { button.textContent = "⎘"; }, 1200);
  }).catch(() => {
    button.textContent = "!";
    setTimeout(() => { button.textContent = "⎘"; }, 1200);
  });
}

function remoteHost() {
  return window.location.hostname || "127.0.0.1";
}

function remoteUser() {
  return document.querySelector("[data-remote-user]")?.dataset.remoteUser || "";
}

function remoteAuthority() {
  const user = remoteUser();
  return `${user ? `${user}@` : ""}${remoteHost()}`;
}

function remotePath(path) {
  return `${remoteAuthority()}:${path.startsWith("/") ? "" : "/"}${path}`;
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function renderOnlineUsers(users) {
  const teamList = document.querySelector("[data-team-user-list]");
  const loginList = document.querySelector("[data-login-user-list]");
  const renderDot = (user) => `<span class="online-user-dot" style="--user-color: ${escapeHtml(user.color)}" aria-hidden="true">${escapeHtml(user.initial)}</span>`;
  let onlineCount = users.length;
  if (teamList) {
    const currentProject = teamList.dataset.currentProject || "";
    const visibleUsers = currentProject
      ? users.filter((user) => user.project === currentProject)
      : users;
    onlineCount = visibleUsers.length;
    teamList.innerHTML = visibleUsers.length
      ? visibleUsers.map((user) => `
        <article class="team-user" title="${escapeHtml(user.name)}">
          ${renderDot(user)}
          <div>
            <strong>${escapeHtml(user.name)}</strong>
            <span>${currentProject ? "正在打开项目" : (user.project ? `参与项目：${escapeHtml(user.project_name || user.project)}` : "未打开项目")}</span>
          </div>
        </article>
      `).join("")
      : `<div class="empty compact" data-team-empty>${currentProject ? "暂无在线用户打开该项目" : "暂无在线用户"}</div>`;
  }
  if (loginList) {
    loginList.innerHTML = users.length
      ? users.map((user) => `
        <div class="online-user" title="${escapeHtml(user.name)}">
          ${renderDot(user)}
          <span class="online-user-text">
            <span class="online-user-name">${escapeHtml(user.name)}</span>
            <span class="online-user-project">${user.project ? `打开项目：${escapeHtml(user.project_name || user.project)}` : "未打开项目"}</span>
          </span>
        </div>
      `).join("")
      : '<p data-login-empty>暂无在线用户</p>';
  }
  document.querySelectorAll("[data-online-count]").forEach((item) => {
    item.textContent = `${onlineCount}`;
  });
}

async function heartbeat() {
  if (!document.querySelector(".username-badge")) {
    return;
  }
  try {
    const response = await fetch("/team/heartbeat", {method: "POST"});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      return;
    }
    renderOnlineUsers(data.users || []);
  } catch (error) {
    // The next heartbeat will retry.
  }
}

heartbeat();
setInterval(heartbeat, 15000);

const teamChat = document.querySelector("[data-team-chat]");
const teamChatMessages = document.querySelector("[data-team-chat-messages]");
const teamChatForm = document.querySelector("[data-team-chat-form]");

function renderTeamChat(messages) {
  if (!teamChat || !teamChatMessages) {
    return;
  }
  const currentUser = teamChat.dataset.currentUser || "";
  teamChatMessages.innerHTML = messages.length
    ? messages.map((message) => `
      <article class="team-chat-message${message.username === currentUser ? " mine" : ""}">
        <span class="online-user-dot" style="--user-color: ${escapeHtml(message.color)}" aria-hidden="true">${escapeHtml(message.initial)}</span>
        <div>
          <div class="team-chat-meta"><strong>${escapeHtml(message.username)}</strong><time>${escapeHtml(message.time)}</time></div>
          <p>${escapeHtml(message.message)}</p>
        </div>
      </article>
    `).join("")
    : '<div class="empty compact" data-team-chat-empty>暂无聊天消息</div>';
  teamChatMessages.scrollTop = teamChatMessages.scrollHeight;
}

async function loadTeamChat() {
  if (!teamChat) {
    return;
  }
  try {
    const response = await fetch("/team/chat", {cache: "no-store"});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      return;
    }
    renderTeamChat(data.messages || []);
  } catch (error) {
    // The next refresh will retry.
  }
}

if (teamChatForm) {
  teamChatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = teamChatForm.elements.message;
    const message = input.value.trim();
    if (!message) {
      input.focus();
      return;
    }
    input.disabled = true;
    try {
      const response = await fetch("/team/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message}),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        alert(data.error || "发送失败");
        return;
      }
      input.value = "";
      renderTeamChat(data.messages || []);
    } catch (error) {
      alert("发送失败");
    } finally {
      input.disabled = false;
      input.focus();
    }
  });
  teamChatForm.elements.message?.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      teamChatForm.requestSubmit();
    }
  });
  loadTeamChat();
  setInterval(loadTeamChat, 3000);
}

if (sftpPanel) {
  const target = sftpPanel.querySelector("[data-sftp-url]");
  const button = sftpPanel.querySelector("[data-copy-sftp]");
  const path = sftpPanel.dataset.sftpPath || "";
  const url = `sftp://${remoteAuthority()}${path.startsWith("/") ? "" : "/"}${path}`;
  if (target) target.textContent = url;
  button?.addEventListener("click", () => {
    copyFeedback(button, url);
  });
}

document.querySelectorAll("[data-rsync-command]").forEach((target) => {
  const container = target.closest("[data-rsync-path]") || target;
  const source = container.dataset.rsyncSource || "./";
  const path = container.dataset.rsyncPath || "";
  const command = `rsync -avz ${source} ${remotePath(path)}`;
  const button = target.closest(".sftp-command-grid, .command-card")?.querySelector("[data-copy-rsync]");
  target.textContent = command;
  button?.addEventListener("click", () => copyFeedback(button, command));
});

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
    window.yoloutilsReloadFooterConsole?.();
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
  setUploadProgress(zone, 0);
  zone.classList.add("uploading");
  zone.classList.remove("upload-complete");

  try {
    const {status, data} = await uploadWithProgress(`/project/${project}/upload/${kind}`, formData, (percent) => {
      setUploadProgress(zone, percent);
    });
    if (status < 200 || status >= 300 || !data.ok) {
      throw new Error(data.error || "上传失败");
    }
    setUploadProgress(zone, 100);
    zone.classList.add("upload-complete");
    if (kind === "images") {
      document.querySelector("[data-image-count]").textContent = `${data.count} 个文件`;
      setAnnotateReady({imagesReady: data.count > 0});
    } else if (kind === "test") {
      document.querySelector("[data-test-count]").textContent = `${data.count} 个文件`;
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
    window.yoloutilsReloadFooterConsole?.();
  } catch (error) {
    alert(error.message);
  } finally {
    window.setTimeout(() => {
      zone.classList.remove("uploading", "upload-complete");
      setUploadProgress(zone, 0);
    }, zone.classList.contains("upload-complete") ? 500 : 0);
  }
}

function setUploadProgress(zone, percent) {
  zone.querySelector(".upload-icon")?.style.setProperty("--upload-progress", `${Math.max(0, Math.min(100, percent))}%`);
}

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }
      onProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("load", () => {
      let data = {};
      try {
        data = JSON.parse(request.responseText || "{}");
      } catch (error) {
        reject(new Error("上传响应解析失败"));
        return;
      }
      resolve({status: request.status, data});
    });
    request.addEventListener("error", () => reject(new Error("上传失败")));
    request.addEventListener("abort", () => reject(new Error("上传已取消")));
    request.send(formData);
  });
}

document.querySelectorAll("[data-upload-zone]").forEach((zone) => {
  const input = zone.querySelector("[data-file-input]") || zone.querySelector("input");
  const directoryInput = zone.querySelector("[data-directory-input]");
  const directoryButton = zone.querySelector("[data-directory-button]");

  zone.addEventListener("click", (event) => {
    if (event.target.closest("input")) {
      return;
    }
    if (event.target.closest("[data-directory-button]")) {
      return;
    }
    input.click();
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    input.click();
  });
  input.addEventListener("change", () => uploadFiles(zone, filesFromFileList(input.files)));
  input.addEventListener("click", () => {
    input.value = "";
  });
  directoryInput?.addEventListener("change", () => uploadFiles(zone, filesFromFileList(directoryInput.files)));
  directoryInput?.addEventListener("click", () => {
    directoryInput.value = "";
  });
  directoryButton?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    directoryInput?.click();
  });
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

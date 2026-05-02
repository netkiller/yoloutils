import os
import json
import shutil
import re
import traceback
from pathlib import Path

from fastapi import Request, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

from workstation import Workstation


templates = Jinja2Templates(directory=TEMPLATES_DIR)
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class SiteWorkstation(Workstation):
    def _directory_tree(self, path: Path):
        tree = super()._directory_tree(path)
        if path == self.workspace and path.name == "images":
            tree["name"] = "根目录"
        return tree


def site_workspace():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else PROJECT_ROOT


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_images_workspace(project: str):
    if not project:
        return None
    workspace = site_workspace()
    project_dir = (workspace / project).resolve()
    if project_dir == workspace or not is_inside(project_dir, workspace):
        return None
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir.resolve()


def write_error_log(workstation: Workstation, error: Exception):
    workspace = workstation.workspace if workstation.workspace else PROJECT_ROOT
    log_file = workspace / ".yoloutils-annotate-error.log"
    try:
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        fallback = PROJECT_ROOT / ".yoloutils-annotate-error.log"
        fallback.write_text(traceback.format_exc(), encoding="utf-8")


def workstation_html(workstation: Workstation, active_mode: str = "annotate"):
    html = workstation._html()
    project_button = (
        '<button id="projectButton" class="header-button" '
        'title="项目" onclick="location.href=\'/project\'">'
        '<span class="header-icon">▤</span><span>项目</span></button>'
    )
    dataset_button = (
        '<button id="createDatasetButton" class="header-button" title="新建数据集">'
        '<span class="header-icon">＋</span><span>新建数据集</span></button>'
    )
    dataset_dialog = """
  <dialog id="datasetDialog" class="dataset-dialog">
    <form id="datasetForm">
      <div class="dataset-dialog-head">
        <h2>新建数据集</h2>
        <button type="button" id="closeDatasetDialog" class="dataset-dialog-close" aria-label="关闭">×</button>
      </div>
      <label><span>数据集名称</span><input name="name" type="text" autocomplete="off" required></label>
      <label><span>val 数量（百分比）</span><input name="val_percent" type="number" min="0" max="100" step="1" value="20" required></label>
      <label><span>test 数量（百分比）</span><input name="test_percent" type="number" min="0" max="100" step="1" value="0" required></label>
      <div class="dataset-dialog-actions">
        <button type="button" id="cancelDatasetDialog">取消</button>
        <button type="submit">确认</button>
      </div>
    </form>
  </dialog>
"""
    dataset_style = """
    .dataset-dialog { width: min(440px, calc(100vw - 32px)); padding: 0; border: 0; border-radius: 8px; box-shadow: 0 24px 64px rgba(31, 41, 51, .26); }
    .dataset-dialog::backdrop { background: rgba(15, 23, 42, .38); }
    .dataset-dialog form { display: grid; gap: 14px; padding: 18px; }
    .dataset-dialog-head, .dataset-dialog-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .dataset-dialog h2 { margin: 0; font-size: 18px; }
    .dataset-dialog label { display: grid; gap: 6px; color: #52606d; font-size: 13px; }
    .dataset-dialog input { width: 100%; height: 36px; padding: 0 10px; border: 1px solid #d9e2ec; border-radius: 6px; font: inherit; }
    .dataset-dialog button { height: 34px; padding: 0 12px; border: 1px solid #d9e2ec; border-radius: 6px; background: #fff; color: #334e68; }
    .dataset-dialog button[type="submit"] { border-color: #2563eb; background: #2563eb; color: #fff; }
    .dataset-dialog-close { width: 32px; padding: 0; font-size: 20px; line-height: 1; }
"""
    dataset_script = """
  <script>
    (() => {
      const dialog = document.getElementById("datasetDialog");
      const openButton = document.getElementById("createDatasetButton");
      const form = document.getElementById("datasetForm");
      const close = () => dialog?.close();
      openButton?.addEventListener("click", () => dialog?.showModal());
      document.getElementById("closeDatasetDialog")?.addEventListener("click", close);
      document.getElementById("cancelDatasetDialog")?.addEventListener("click", close);
      form?.addEventListener("submit", async (event) => {
        event.preventDefault();
        const data = Object.fromEntries(new FormData(form).entries());
        const response = await fetch("/annotate/api/datasets", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(data),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !payload.ok) {
          alert(payload.error || "创建数据集失败");
          return;
        }
        location.href = "/dataset";
      });
    })();
  </script>
"""
    html = (
        html.replace('"/api/', '"/annotate/api/')
        .replace("'/api/", "'/annotate/api/")
        .replace("`/api/", "`/annotate/api/")
        .replace('"/media', '"/annotate/media')
        .replace("'/media", "'/annotate/media")
        .replace("`/media", "`/annotate/media")
        .replace(
            '<button id="annotateModeButton"',
            f'{project_button}<button id="annotateModeButton"',
            1,
        )
        .replace(
            '<button id="autoAnnotate"',
            f'{dataset_button}<button id="autoAnnotate"',
            1,
        )
        .replace(
            'id="annotateModeButton" class="header-button active"',
            'id="annotateModeButton" class="header-button" onclick="location.href=\'/annotate\'"',
        )
        .replace(
            'id="datasetButton" class="header-button"',
            'id="datasetButton" class="header-button" onclick="location.href=\'/dataset\'"',
        )
        .replace(
            'id="trainButton" class="header-button"',
            'id="trainButton" class="header-button" onclick="location.href=\'/train\'"',
        )
        .replace(
            'datasetButton.addEventListener("click", showEnterpriseNotice);',
            'datasetButton.addEventListener("click", () => { location.href = "/dataset"; });',
        )
        .replace(
            'trainButton.addEventListener("click", showEnterpriseNotice);',
            'trainButton.addEventListener("click", () => { location.href = "/train"; });',
        )
        .replace(
            '<button id="trainButton" class="header-button" onclick="location.href=\'/train\'" title="训练"><span class="header-icon">▶</span><span>训练</span></button>',
            '<button id="trainButton" class="header-button" onclick="location.href=\'/train\'" title="训练"><span class="header-icon">▶</span><span>训练</span></button>'
            '<button id="validateButton" class="header-button" title="验证" onclick="location.href=\'/validate\'"><span class="header-icon">✓</span><span>验证</span></button>',
            1,
        )
    )
    active_button = {
        "annotate": "annotateModeButton",
        "dataset": "datasetButton",
        "train": "trainButton",
        "validate": "validateButton",
    }.get(active_mode)
    if active_button:
        html = html.replace(
            f'id="{active_button}" class="header-button"',
            f'id="{active_button}" class="header-button active"',
        )
    return html.replace("</style>", f"{dataset_style}</style>", 1).replace(
        "</body>",
        f"{dataset_dialog}{dataset_script}</body>",
        1,
    )


def create_workstation():
    workstation = SiteWorkstation()
    dataset = os.environ.get("YOLOUTILS_DATASET")
    run = os.environ.get("YOLOUTILS_RUN")

    workstation.workspace = site_workspace()
    workstation.dataset = Path(dataset).expanduser().resolve() if dataset else None
    workstation.run = Path(run).expanduser().resolve() if run else None
    workstation.requested_classes_file = os.environ.get("YOLOUTILS_CLASSES") or None
    workstation.open_browser = False
    workstation.team_mode = os.environ.get("YOLOUTILS_TEAM", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    workstation.mdns = workstation._normalize_mdns(
        os.environ.get("YOLOUTILS_MDNS", "netkiller.local")
    )
    workstation.class_groups = workstation._load_class_groups()
    workstation.classes_file = (
        workstation.class_groups[0]["path"] if workstation.class_groups else None
    )
    workstation.classes = (
        workstation.class_groups[0]["classes"] if workstation.class_groups else []
    )
    return workstation


def current_project_dir(workstation: Workstation):
    workspace = workstation.workspace.resolve()
    if workspace.name == "images" and is_inside(workspace.parent, site_workspace()):
        return workspace.parent
    return workspace


def image_files(workstation: Workstation, root: Path):
    return sorted(
        (path for path in root.rglob("*") if workstation._is_image(path)),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def copy_image_with_label(source: Path, source_root: Path, target_root: Path):
    relative = source.relative_to(source_root)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    label = source.with_suffix(".txt")
    if label.is_file():
        label_target = target.with_suffix(".txt")
        shutil.copy2(label, label_target)


def build_dataset(workstation: Workstation, name: str, val_percent: int, test_percent: int):
    name = (name or "").strip()
    if not name or not DATASET_NAME_PATTERN.match(name):
        return None, "数据集名称只能包含字母、数字、点、下划线和连字符"
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
        return None, "val 和 test 百分比之和不能超过 100"

    images_root = workstation.workspace.resolve()
    project_dir = current_project_dir(workstation)
    dataset_dir = project_dir / "datasets" / name
    if dataset_dir.exists():
        return None, "数据集已存在"

    files = image_files(workstation, images_root)
    total = len(files)
    test_count = round(total * test_percent / 100)
    val_count = round(total * val_percent / 100)
    test_files = files[:test_count]
    val_files = files[test_count : test_count + val_count]
    train_files = files[test_count + val_count :]

    for split, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source in split_files:
            copy_image_with_label(source, images_root, split_dir)

    return {
        "path": str(dataset_dir),
        "total": total,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }, None


def apply_project_workspace(workstation: Workstation, project: str):
    images_dir = project_images_workspace(project)
    workstation.workspace = images_dir if images_dir is not None else site_workspace()
    workstation.class_groups = workstation._load_class_groups()
    workstation.classes_file = (
        workstation.class_groups[0]["path"] if workstation.class_groups else None
    )
    workstation.classes = (
        workstation.class_groups[0]["classes"] if workstation.class_groups else []
    )


def create_annotate_app():
    workstation = create_workstation()
    app = workstation._create_app()
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == "/"
            and "GET" in getattr(route, "methods", set())
        )
    ]

    @app.get("/")
    def index(request: Request):
        try:
            apply_project_workspace(workstation, request.query_params.get("project", ""))
            return HTMLResponse(workstation_html(workstation, "annotate"))
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.post("/api/datasets")
    async def create_dataset(request: Request):
        try:
            payload = await request.json()
            result, error = build_dataset(
                workstation,
                str(payload.get("name", "")),
                int(payload.get("val_percent", 0) or 0),
                int(payload.get("test_percent", 0) or 0),
            )
            if error:
                return JSONResponse({"ok": False, "error": error}, status_code=400)
            return {"ok": True, **result}
        except Exception as error:
            write_error_log(workstation, error)
            return JSONResponse(
                {"ok": False, "error": "创建数据集失败"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app

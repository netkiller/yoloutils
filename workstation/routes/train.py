import json
import os
import csv
import base64
import shlex
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
queue_lock = threading.Lock()
worker_thread = None
running_processes = {}
MODEL_VERSIONS = [f"YOLOv{number}" for number in range(3, 13)] + ["YOLO26"]
MODEL_SIZES = ["N", "S", "M", "L", "X"]
COMPLETE_STATUS = "完成"
ACTIVE_STATUSES = {"排队中", "进行中"}
WEIGHT_FILES = {"best.pt", "last.pt"}
RESULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def clean_model_version(value: str):
    value = (value or "").strip()
    return value if value in MODEL_VERSIONS else "YOLO26"


def clean_model_size(value: str):
    value = (value or "").strip().upper()
    return value if value in MODEL_SIZES else "N"


def model_weight(version: str, size: str):
    suffix = clean_model_size(size).lower()
    version = clean_model_version(version)
    if version == "YOLO26":
        return f"yolo26{suffix}.pt"
    return f"yolov{version.removeprefix('YOLOv')}{suffix}.pt"


def optional_int(value: str):
    value = (value or "").strip()
    return int(value) if value else None


def display_datetime(value: str):
    value = (value or "").strip()
    if not value:
        return ""
    return value.replace("T", " ")[:16]


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def queue_dir():
    path = workspace_path() / ".train"
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def tasks_file():
    return queue_dir() / "tasks.json"


def load_tasks():
    path = tasks_file()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_tasks(tasks):
    tasks_file().write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def update_task(task_id, **updates):
    with queue_lock:
        tasks = load_tasks()
        for task in tasks:
            if task["id"] == task_id:
                task.update(updates)
                break
        save_tasks(tasks)


def project_dirs():
    workspace = workspace_path()
    if not workspace.is_dir():
        return []
    return [path for path in sorted(workspace.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]


def dataset_dirs(project: str = ""):
    datasets = []
    for project_dir in project_dirs():
        if project and project_dir.name != project:
            continue
        root = project_dir / "datasets"
        if not root.is_dir():
            continue
        for dataset_dir in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if dataset_dir.is_dir():
                datasets.append({"project": project_dir.name, "name": dataset_dir.name, "path": dataset_dir})
    return datasets


def selected_dataset(project: str, dataset: str):
    for item in dataset_dirs(project):
        if item["name"] == dataset:
            return item
    items = dataset_dirs(project)
    return items[0] if items else None


def read_classes(project: str):
    candidates = [
        workspace_path() / project / "classes.txt",
        workspace_path() / "classes.txt",
    ]
    for path in candidates:
        if path.is_file():
            classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if classes:
                return classes
    return ["object"]


def write_data_yaml(task):
    dataset_path = Path(task["dataset_path"])
    classes = read_classes(task["project"])
    yaml_path = queue_dir() / f"{task['id']}.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_path}",
                "train: train",
                "val: val",
                "test: test",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def append_log(task_id, text):
    log_file(task_id).open("a", encoding="utf-8").write(text)


def log_file(task_id):
    return queue_dir() / "logs" / f"{task_id}.log"


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_runs_dir(project: str):
    return workspace_path() / project / "runs"


def display_project_name(workspace: Path, project: str):
    project = (project or "").strip()
    if not project:
        return ""
    path = workspace / project
    meta = path / ".project"
    if not meta.is_file():
        return project
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return str(data.get("name") or project)
    except (OSError, ValueError, TypeError):
        return project


def expected_run_dir(task):
    return project_runs_dir(task["project"]) / task["name"]


def task_run_dir(task):
    stored = task.get("run_path")
    if stored:
        path = Path(stored).expanduser().resolve()
        runs = project_runs_dir(task["project"]).resolve()
        if is_inside(path, runs) and path.is_dir():
            return path

    expected = expected_run_dir(task)
    if expected.is_dir():
        return expected.resolve()

    runs = project_runs_dir(task["project"])
    if not runs.is_dir():
        return expected.resolve()
    candidates = [path for path in runs.iterdir() if path.is_dir() and path.name.startswith(task["name"])]
    if not candidates:
        return expected.resolve()
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def encode_run_id(project: str, run_name: str):
    raw = f"{project}/{run_name}".encode("utf-8")
    return "run-" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_run_id(model_id: str):
    if not model_id.startswith("run-"):
        return None
    payload = model_id.removeprefix("run-")
    payload += "=" * (-len(payload) % 4)
    try:
        project, run_name = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8").split("/", 1)
    except (ValueError, UnicodeDecodeError):
        return None
    return project, run_name


def synthetic_run_task(project: str, run_dir: Path):
    return {
        "id": encode_run_id(project, run_dir.name),
        "name": run_dir.name,
        "project": project,
        "dataset": "runs",
        "model": "",
        "epochs": "",
        "status": COMPLETE_STATUS,
        "run_path": str(run_dir.resolve()),
        "synthetic_run": True,
    }


def model_items(tasks, current_project: str = ""):
    items = []
    seen_runs = set()
    for task in tasks:
        if task.get("status") != COMPLETE_STATUS:
            continue
        run_dir = task_run_dir(task)
        if run_dir.is_dir():
            seen_runs.add(run_dir.resolve())
        weights_dir = run_dir / "weights"
        items.append(
            {
                "task": task,
                "run_dir": run_dir,
                "has_best": (weights_dir / "best.pt").is_file(),
                "has_last": (weights_dir / "last.pt").is_file(),
                "results_image": f"/train/models/{task['id']}/files/results.png" if (run_dir / "results.png").is_file() else "",
                "metrics": run_metrics_summary(run_dir),
                "summary": f"{task.get('project', '')} / {task.get('dataset', '')}",
                "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
                if run_dir.is_dir()
                else task.get("finished_at", task.get("created_at", "")),
                "display_time": display_datetime(task.get("finished_at") or (
                    datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
                    if run_dir.is_dir()
                    else task.get("created_at", "")
                )),
            }
        )
    for project_dir in project_dirs():
        if current_project and project_dir.name != current_project:
            continue
        runs_dir = project_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda item: item.name.lower()):
            resolved = run_dir.resolve()
            if resolved in seen_runs:
                continue
            task = synthetic_run_task(project_dir.name, run_dir)
            weights_dir = run_dir / "weights"
            items.append(
                {
                    "task": task,
                    "run_dir": resolved,
                    "has_best": (weights_dir / "best.pt").is_file(),
                    "has_last": (weights_dir / "last.pt").is_file(),
                    "results_image": f"/train/models/{task['id']}/files/results.png" if (run_dir / "results.png").is_file() else "",
                    "metrics": run_metrics_summary(run_dir),
                    "summary": "",
                    "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                    "display_time": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def filtered_queue_tasks(tasks, queue_filter: str):
    if queue_filter == "completed":
        return [task for task in tasks if task.get("status") == COMPLETE_STATUS]
    if queue_filter == "active":
        return [task for task in tasks if task.get("status") != COMPLETE_STATUS]
    return tasks


def read_text_file(path: Path, max_chars: int = 12000):
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def read_results_csv(path: Path, max_rows: int = 80):
    if not path.is_file():
        return {"headers": [], "rows": []}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return {"headers": [], "rows": []}
    return {"headers": rows[0], "rows": rows[1:max_rows + 1]}


def parse_args_text(text: str):
    values = {}
    for line in text.splitlines():
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        values[key] = value
    return values


def command_value(value: str):
    value = (value or "").strip()
    if value.lower() in {"", "null", "none"}:
        return ""
    return shlex.quote(value)


def training_command_from_args(text: str):
    values = parse_args_text(text)
    task = values.get("task") or "detect"
    mode = values.get("mode") or "train"
    parts = ["yolo", task, mode]
    keys = ["model", "data", "epochs", "batch", "imgsz", "device", "workers", "project", "name", "amp"]
    for key in keys:
        value = command_value(values.get(key, ""))
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_metric(value: str):
    value = (value or "").strip()
    if not value:
        return "-"
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def run_metrics_summary(run_dir: Path):
    path = run_dir / "results.csv"
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row]
    if not rows:
        return ""
    last = rows[-1]
    epoch = (last.get("epoch") or "-").strip()
    precision = format_metric(last.get("metrics/precision(B)") or last.get("metrics/precision"))
    recall = format_metric(last.get("metrics/recall(B)") or last.get("metrics/recall"))
    map50 = format_metric(last.get("metrics/mAP50(B)") or last.get("metrics/mAP50"))
    map5095 = format_metric(last.get("metrics/mAP50-95(B)") or last.get("metrics/mAP50-95"))
    return [
        {"label": "Epochs", "value": epoch},
        {"label": "mAP50", "value": map50},
        {"label": "mAP50-95", "value": map5095},
        {"label": "Precision", "value": precision},
        {"label": "Recall", "value": recall},
    ]


def run_result_assets(task):
    run_dir = task_run_dir(task)
    if not run_dir.is_dir():
        return {
            "run_dir": run_dir,
            "has_best": False,
            "has_last": False,
            "args": "",
            "train_command": "",
            "csv": {"headers": [], "rows": []},
            "metrics": [],
            "images": [],
            "image_tabs": [],
            "files": [],
        }

    args = read_text_file(run_dir / "args.yaml") or read_text_file(run_dir / "args.json") or read_text_file(run_dir / "args.txt")
    images = []
    image_tabs = {
        "results": {"label": "results.png", "images": []},
        "labels": {"label": "labels.jpg", "images": []},
        "curve": {"label": "曲线", "images": []},
        "train": {"label": "train", "images": []},
        "val": {"label": "val", "images": []},
        "confusion": {"label": "confusion", "images": []},
    }
    files = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.relative_to(run_dir).as_posix().lower()):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if path.suffix.lower() in RESULT_IMAGE_EXTS:
            image = {"name": relative, "src": f"/train/models/{task['id']}/files/{relative}"}
            images.append(image)
            filename = path.name.lower()
            if filename.startswith("train_") and path.suffix.lower() in {".jpg", ".jpeg"}:
                image_tabs["train"]["images"].append(image)
            elif filename.startswith("val_") and path.suffix.lower() in {".jpg", ".jpeg"}:
                image_tabs["val"]["images"].append(image)
            elif filename == "results.png":
                image_tabs["results"]["images"].append(image)
            elif filename == "labels.jpg":
                image_tabs["labels"]["images"].append(image)
            elif filename.endswith("_curve.png"):
                image_tabs["curve"]["images"].append(image)
            elif filename.startswith("confusion") and filename.endswith(".png"):
                image_tabs["confusion"]["images"].append(image)
        elif path.name not in WEIGHT_FILES:
            files.append({"name": relative, "size": path.stat().st_size, "href": f"/train/models/{task['id']}/files/{relative}"})
    return {
        "run_dir": run_dir,
        "has_best": (run_dir / "weights" / "best.pt").is_file(),
        "has_last": (run_dir / "weights" / "last.pt").is_file(),
        "args": args,
        "train_command": training_command_from_args(args) if args else "",
        "csv": read_results_csv(run_dir / "results.csv"),
        "metrics": run_metrics_summary(run_dir),
        "images": images,
        "image_tabs": [item for item in image_tabs.values() if item["images"]],
        "files": files,
    }


def train_command(task):
    data_value = task.get("data") or str(write_data_yaml(task))
    command = [
        "yolo",
        "detect",
        "train",
        f"data={data_value}",
        f"model={task['model']}",
        f"epochs={task['epochs']}",
        f"project={project_runs_dir(task['project'])}",
        f"name={task['name']}",
    ]
    for key in ("imgsz", "batch", "device", "workers", "amp"):
        value = task.get(key)
        if value not in (None, ""):
            command.append(f"{key}={value}")
    return command


def run_task(task):
    task_id = task["id"]
    run_dir = expected_run_dir(task)
    update_task(
        task_id,
        status="进行中",
        run_path=str(run_dir),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    command = train_command(task)
    append_log(task_id, "$ " + " ".join(str(part) for part in command) + "\n\n")
    if shutil.which("yolo") is None:
        append_log(task_id, "yolo 命令不存在，请先安装 ultralytics 或确认虚拟环境 PATH。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    try:
        with log_file(task_id).open("a", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                cwd=workspace_path(),
                text=True,
            )
            running_processes[task_id] = process
            return_code = process.wait()
    except Exception as error:
        append_log(task_id, f"\n训练启动失败: {error}\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    finally:
        running_processes.pop(task_id, None)

    status_text = "完成" if return_code == 0 else "失败"
    append_log(task_id, f"\n进程退出码: {return_code}\n")
    update_task(task_id, status=status_text, finished_at=datetime.now().isoformat(timespec="seconds"))


def worker_loop():
    while True:
        with queue_lock:
            tasks = load_tasks()
            task = next((item for item in tasks if item["status"] == "排队中"), None)
        if task is None:
            return
        run_task(task)


def ensure_worker():
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    worker_thread = threading.Thread(target=worker_loop, daemon=True, name="yoloutils-train-worker")
    worker_thread.start()


async def form_fields(request: Request):
    body = (await request.body()).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


@router.get("/train")
def train(request: Request, project: str = "", tab: str = "models", queue: str = "active"):
    workspace = workspace_path()
    current_project = project or request.cookies.get("current_project", "")
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    if current_project:
        tasks = [task for task in tasks if task.get("project") == current_project]
    active_tab = tab if tab in {"models", "queue"} else "models"
    queue_filter = queue if queue in {"active", "completed", "all"} else "active"
    response = templates.TemplateResponse(
        request=request,
            name="train/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "tasks": tasks,
            "queue_tasks": filtered_queue_tasks(tasks, queue_filter),
            "models": model_items(tasks, current_project),
            "active_tab": active_tab,
            "queue_filter": queue_filter,
            "active_page": "train",
            "current_project": current_project,
            "current_project_name": display_project_name(workspace, current_project),
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


def find_task(task_id: str):
    return next((item for item in load_tasks() if item["id"] == task_id), None)


def resolve_model_task(model_id: str):
    task = find_task(model_id)
    if task is not None:
        return task
    decoded = decode_run_id(model_id)
    if decoded is None:
        return None
    project, run_name = decoded
    runs_dir = project_runs_dir(project).resolve()
    run_dir = (runs_dir / run_name).resolve()
    if not is_inside(run_dir, runs_dir) or not run_dir.is_dir():
        return None
    return synthetic_run_task(project, run_dir)


@router.get("/train/models/{task_id}")
def train_model(request: Request, task_id: str):
    workspace = workspace_path()
    task = resolve_model_task(task_id)
    if task is None:
        return RedirectResponse(url="/train", status_code=status.HTTP_303_SEE_OTHER)
    current_project = task.get("project", request.cookies.get("current_project", ""))
    assets = run_result_assets(task)
    return templates.TemplateResponse(
        request=request,
        name="train/model.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": task,
            "assets": assets,
            "active_page": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )


@router.get("/train/models/{task_id}/weights/{weight_name}")
def download_model_weight(task_id: str, weight_name: str):
    task = resolve_model_task(task_id)
    if task is None or weight_name not in WEIGHT_FILES:
        return JSONResponse({"ok": False, "error": "模型不存在"}, status_code=404)
    run_dir = task_run_dir(task)
    path = (run_dir / "weights" / weight_name).resolve()
    weights_dir = (run_dir / "weights").resolve()
    if not is_inside(path, weights_dir) or not path.is_file():
        return JSONResponse({"ok": False, "error": "模型文件不存在"}, status_code=404)
    return FileResponse(path, filename=f"{task['name']}-{weight_name}")


@router.get("/train/models/{task_id}/files/{file_path:path}")
def train_model_file(task_id: str, file_path: str):
    task = resolve_model_task(task_id)
    if task is None:
        return JSONResponse({"ok": False, "error": "文件不存在"}, status_code=404)
    run_dir = task_run_dir(task).resolve()
    path = (run_dir / file_path).resolve()
    if not is_inside(path, run_dir) or not path.is_file():
        return JSONResponse({"ok": False, "error": "文件不存在"}, status_code=404)
    return FileResponse(path)


@router.get("/train/new")
def new_train(request: Request, project: str = "", dataset: str = ""):
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    project = project or current_project
    current_project = project
    dataset_item = selected_dataset(project, dataset)
    response = templates.TemplateResponse(
        request=request,
        name="train/new.html",
        context={
            "request": request,
            "workspace": workspace,
            "dataset": dataset_item,
            "datasets": dataset_dirs(project),
            "model_versions": MODEL_VERSIONS,
            "model_sizes": MODEL_SIZES,
            "default_model_version": "YOLO26",
            "default_model_size": "N",
            "active_page": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if project:
        response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/train")
async def create_train(request: Request):
    form = await form_fields(request)
    project = form.get("project", [""])[0]
    dataset = form.get("dataset", [""])[0]
    dataset_item = selected_dataset(project, dataset)
    if dataset_item is None:
        return RedirectResponse(url="/train/new", status_code=status.HTTP_303_SEE_OTHER)

    model_version = clean_model_version(form.get("model_version", ["YOLO26"])[0])
    model_size = clean_model_size(form.get("model_size", ["N"])[0])
    data_value = form.get("data", [""])[0].strip()
    task = {
        "id": uuid4().hex[:12],
        "name": form.get("name", ["train"])[0].strip() or "train",
        "project": dataset_item["project"],
        "dataset": dataset_item["name"],
        "dataset_path": str(dataset_item["path"]),
        "model_version": model_version,
        "model_size": model_size,
        "model": model_weight(model_version, model_size),
        "epochs": int(form.get("epochs", ["200"])[0] or 200),
        "imgsz": optional_int(form.get("imgsz", [""])[0]),
        "batch": optional_int(form.get("batch", [""])[0]),
        "device": form.get("device", [""])[0].strip(),
        "workers": optional_int(form.get("workers", [""])[0]),
        "amp": form.get("amp", [""])[0].strip(),
        "data": data_value,
        "status": "排队中",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with queue_lock:
        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
    append_log(task["id"], f"任务已创建: {task['created_at']}\n")
    ensure_worker()
    return RedirectResponse(url="/train", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/train/tasks/{task_id}")
def train_task(request: Request, task_id: str):
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    tasks = load_tasks()
    task = next((item for item in tasks if item["id"] == task_id), None)
    if task is None:
        return RedirectResponse(url="/train", status_code=status.HTTP_303_SEE_OTHER)
    log = log_file(task_id).read_text(encoding="utf-8", errors="replace") if log_file(task_id).is_file() else ""
    return templates.TemplateResponse(
        request=request,
        name="train/task.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": task,
            "log": log,
            "active_page": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )


@router.get("/train/tasks/{task_id}/logs")
def train_task_logs(task_id: str):
    tasks = load_tasks()
    task = next((item for item in tasks if item["id"] == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "任务不存在"}, status_code=404)
    path = log_file(task_id)
    log = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        "ok": True,
        "task": task,
        "log": log,
        "size": path.stat().st_size if path.is_file() else 0,
    }


@router.post("/train/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    process = running_processes.get(task_id)
    if process and process.poll() is None:
        process.terminate()
    update_task(task_id, status="取消", finished_at=datetime.now().isoformat(timespec="seconds"))
    append_log(task_id, "\n任务已取消。\n")
    return RedirectResponse(url="/train", status_code=status.HTTP_303_SEE_OTHER)

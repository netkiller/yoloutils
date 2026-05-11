import csv
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel"}
queue_lock = threading.Lock()
worker_thread = None
running_processes = {}


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_path(workspace: Path, project: str):
    if not project:
        return None
    path = (workspace / project).resolve()
    if path == workspace or not is_inside(path, workspace):
        return None
    return path if path.is_dir() else None


def read_project_name(path: Path):
    meta_file = path / ".project"
    if not meta_file.is_file():
        return path.name
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return str(data.get("name") or path.name)
    except (OSError, json.JSONDecodeError):
        return path.name


def dataset_items(path: Path):
    datasets_dir = path / "datasets"
    if not datasets_dir.is_dir():
        return []
    return [
        {"name": item.name, "path": item}
        for item in sorted(datasets_dir.iterdir(), key=lambda dataset: dataset.name.lower())
        if item.is_dir()
    ]


def queue_dir():
    path = workspace_path() / ".validate"
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


def append_log(task_id, text):
    log_file(task_id).open("a", encoding="utf-8").write(text)


def log_file(task_id):
    return queue_dir() / "logs" / f"{task_id}.log"


def metric_value(row: dict, keywords):
    for key, value in row.items():
        key_text = key.lower().replace(" ", "")
        if all(keyword in key_text for keyword in keywords):
            try:
                return round(float(value), 4)
            except (TypeError, ValueError):
                return None
    return None


def read_metrics(run_dir: Path):
    results_file = run_dir / "results.csv"
    if not results_file.is_file():
        return {}
    try:
        with results_file.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {
        "precision": metric_value(row, ["precision"]),
        "recall": metric_value(row, ["recall"]),
        "map50": metric_value(row, ["map50"]),
        "map5095": metric_value(row, ["map50-95"]),
    }


def model_item(file: Path, root: Path, source: str):
    stat = file.stat()
    run_dir = file.parent.parent if file.parent.name == "weights" else file.parent
    metrics = read_metrics(run_dir)
    return {
        "name": file.stem,
        "filename": file.name,
        "path": file,
        "relative_path": file.relative_to(root).as_posix(),
        "source": source,
        "run": run_dir.name if source == "训练" else "",
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics,
    }


def model_items(path: Path):
    models = []
    seen = set()
    sources = [
        (path / "train-runs", "训练"),
        (path / "models", "上传"),
    ]
    for root, source in sources:
        if not root.is_dir():
            continue
        for file in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not file.is_file() or file.suffix.lower() not in MODEL_EXTS:
                continue
            resolved = file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(model_item(file, path, source))
    return sorted(models, key=lambda item: item["path"].stat().st_mtime, reverse=True)


def read_classes(path: Path):
    candidates = [path / "classes.txt", workspace_path() / "classes.txt"]
    for file in candidates:
        if file.is_file():
            classes = [line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if classes:
                return classes
    return ["object"]


def write_data_yaml(task):
    project = project_path(workspace_path(), task["project"])
    dataset_path = project / "datasets" / task["dataset"]
    classes = read_classes(project)
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


def validate_command(task):
    project = project_path(workspace_path(), task["project"])
    model_path = (project / task["model"]).resolve()
    data_yaml = write_data_yaml(task)
    return [
        "yolo",
        "detect",
        "val",
        f"data={data_yaml}",
        f"model={model_path}",
        f"split={task['split']}",
        f"imgsz={task['imgsz']}",
        f"project={project / 'validate-runs'}",
        f"name={task['name']}",
    ] + ([f"device={task['device']}"] if task.get("device") else [])


def run_task(task):
    task_id = task["id"]
    update_task(task_id, status="进行中", started_at=datetime.now().isoformat(timespec="seconds"))
    command = validate_command(task)
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
        append_log(task_id, f"\n任务启动失败: {error}\n")
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
    worker_thread = threading.Thread(target=worker_loop, daemon=True, name="yoloutils-validate-worker")
    worker_thread.start()


async def form_fields(request: Request):
    body = (await request.body()).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


def project_tasks(project: str):
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    return [task for task in tasks if task.get("project") == project]


@router.get("/validate")
def validate(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/validate/{project}", status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    path = project_path(workspace, current_project)
    models = model_items(path) if path else []
    datasets = dataset_items(path) if path else []
    tasks = project_tasks(current_project) if current_project else []
    response = templates.TemplateResponse(
        request=request,
        name="validate/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "validate",
            "current_project": current_project,
            "project_name": read_project_name(path) if path else "",
            "models": models,
            "datasets": datasets,
            "tasks": tasks,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/validate/{project}")
def validate_with_project(request: Request, project: str):
    workspace = workspace_path()
    current_project = project
    path = project_path(workspace, current_project)
    models = model_items(path) if path else []
    datasets = dataset_items(path) if path else []
    tasks = project_tasks(current_project) if current_project else []
    response = templates.TemplateResponse(
        request=request,
        name="validate/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "validate",
            "current_project": current_project,
            "project_name": read_project_name(path) if path else "",
            "models": models,
            "datasets": datasets,
            "tasks": tasks,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.post("/validate/run")
async def create_validate_task(request: Request):
    form = await form_fields(request)
    project = form.get("project", [""])[0] or request.cookies.get("current_project", "")
    path = project_path(workspace_path(), project)
    if path is None:
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)

    model = form.get("model", [""])[0]
    dataset = form.get("dataset", [""])[0]
    split = form.get("split", ["val"])[0]
    if split not in {"val", "test"}:
        split = "val"
    model_path = (path / model).resolve()
    dataset_path = (path / "datasets" / dataset).resolve()
    datasets_root = (path / "datasets").resolve()
    if (
        not model
        or not dataset
        or not is_inside(model_path, path)
        or not model_path.is_file()
        or dataset_path == datasets_root
        or not is_inside(dataset_path, datasets_root)
        or not dataset_path.is_dir()
    ):
        return RedirectResponse(url=f"/validate/{project}", status_code=status.HTTP_303_SEE_OTHER)

    default_name = f"{split}-{dataset}-{model_path.stem}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        imgsz = int(form.get("imgsz", ["640"])[0] or 640)
    except ValueError:
        imgsz = 640
    task = {
        "id": uuid4().hex[:12],
        "name": form.get("name", [default_name])[0].strip() or default_name,
        "project": project,
        "dataset": dataset,
        "model": model,
        "split": split,
        "mode": "测试" if split == "test" else "检验",
        "imgsz": imgsz,
        "device": form.get("device", [""])[0].strip(),
        "status": "排队中",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with queue_lock:
        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
    append_log(task["id"], f"任务已创建: {task['created_at']}\n")
    ensure_worker()
    return RedirectResponse(url=f"/validate/tasks/{task['id']}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/validate/tasks/{task_id}")
def validate_task(request: Request, task_id: str):
    current_project = request.cookies.get("current_project", "")
    task = next((item for item in load_tasks() if item["id"] == task_id), None)
    if task is None:
        return RedirectResponse(url="/validate", status_code=status.HTTP_303_SEE_OTHER)
    log = log_file(task_id).read_text(encoding="utf-8", errors="replace") if log_file(task_id).is_file() else ""
    return templates.TemplateResponse(
        request=request,
        name="validate/task.html",
        context={
            "request": request,
            "workspace": workspace_path(),
            "task": task,
            "log": log,
            "active_page": "validate",
            "current_project": current_project or task.get("project", ""),
            **header_context(request, workspace_path()),
        },
    )


@router.post("/validate/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    process = running_processes.get(task_id)
    if process and process.poll() is None:
        process.terminate()
    update_task(task_id, status="取消", finished_at=datetime.now().isoformat(timespec="seconds"))
    append_log(task_id, "\n任务已取消。\n")
    return RedirectResponse(url="/validate", status_code=status.HTTP_303_SEE_OTHER)

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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
queue_lock = threading.Lock()
worker_thread = None
running_processes = {}


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


def train_command(task):
    data_yaml = write_data_yaml(task)
    return [
        "yolo",
        "detect",
        "train",
        f"data={data_yaml}",
        f"model={task['model']}",
        f"epochs={task['epochs']}",
        f"imgsz={task['imgsz']}",
        f"batch={task['batch']}",
        f"project={workspace_path() / task['project'] / 'train-runs'}",
        f"name={task['name']}",
    ] + ([f"device={task['device']}"] if task.get("device") else [])


def run_task(task):
    task_id = task["id"]
    update_task(task_id, status="进行中", started_at=datetime.now().isoformat(timespec="seconds"))
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
def train(request: Request, project: str = ""):
    workspace = workspace_path()
    current_project = project or request.cookies.get("current_project", "")
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    if current_project:
        tasks = [task for task in tasks if task.get("project") == current_project]
    response = templates.TemplateResponse(
        request=request,
            name="train/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "tasks": tasks,
            "active_page": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


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

    task = {
        "id": uuid4().hex[:12],
        "name": form.get("name", ["train"])[0].strip() or "train",
        "project": dataset_item["project"],
        "dataset": dataset_item["name"],
        "dataset_path": str(dataset_item["path"]),
        "model": form.get("model", ["yolov8n.pt"])[0].strip() or "yolov8n.pt",
        "epochs": int(form.get("epochs", ["200"])[0] or 200),
        "imgsz": int(form.get("imgsz", ["640"])[0] or 640),
        "batch": int(form.get("batch", ["16"])[0] or 16),
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

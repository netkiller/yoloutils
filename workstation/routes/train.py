import json
import os
import csv
import base64
import posixpath
import shlex
import shutil
import stat as stat_module
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from routes.dataset import count_dataset_split, read_deploy_tasks
from routes.project import header_context
from routes.resources import find_resource, ssh_connect_kwargs


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
MODEL_EXPORT_FORMATS = {
    "onnx": "ONNX",
    "torchscript": "TorchScript",
    "openvino": "OpenVINO",
    "engine": "TensorRT",
    "coreml": "CoreML",
    "tflite": "TFLite",
    "ncnn": "NCNN",
}
YOLO_OPTION_KEYS = (
    "imgsz",
    "batch",
    "device",
    "workers",
    "amp",
    "data",
    "patience",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
    "close_mosaic",
    "cache",
    "rect",
    "resume",
    "pretrained",
    "seed",
    "deterministic",
    "single_cls",
    "plots",
    "save_period",
    "freeze",
    "dropout",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "erasing",
    "crop_fraction",
)


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


def optional_value(value: str):
    value = (value or "").strip()
    return value if value else None


def display_datetime(value: str):
    value = (value or "").strip()
    if not value:
        return ""
    return value.replace("T", " ")[:16]


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def demo_mode_enabled():
    return os.environ.get("YOLOUTILS_DEMO") == "1"


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


def matched_dataset(project: str, dataset: str):
    dataset = (dataset or "").strip()
    if not dataset:
        return None
    for item in dataset_dirs(project):
        if item["name"] == dataset:
            return item
    return None


def remote_dataset_dirs(project: str = ""):
    items = []
    for project_dir in project_dirs():
        if project and project_dir.name != project:
            continue
        for task in read_deploy_tasks(project_dir):
            if task.get("status") != COMPLETE_STATUS:
                continue
            if task.get("target_type") != "remote":
                continue
            dataset_name = str(task.get("dataset") or "")
            if not dataset_name:
                continue
            resource_id = str(task.get("resource_id") or "")
            remote_path = str(task.get("target_path") or "")
            if not resource_id or not remote_path:
                continue
            items.append(
                {
                    "key": f"{resource_id}:{dataset_name}:{remote_path}",
                    "project": project_dir.name,
                    "name": dataset_name,
                    "path": Path(str(task.get("source_path") or project_dir / "datasets" / dataset_name)),
                    "remote_path": remote_path,
                    "resource_id": resource_id,
                    "resource_name": str(task.get("resource_name") or "算力服务器"),
                }
            )
    return items


def selected_remote_dataset(project: str, dataset: str):
    items = remote_dataset_dirs(project)
    for item in items:
        if item["name"] == dataset or item.get("key") == dataset:
            return item
    return items[0] if items else None


def matched_remote_dataset(project: str, dataset: str):
    dataset = (dataset or "").strip()
    if not dataset:
        return None
    for item in remote_dataset_dirs(project):
        if item["name"] == dataset or item.get("key") == dataset:
            return item
    return None


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


def data_yaml_text(task, dataset_path: str):
    classes = read_classes(task["project"])
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    return "\n".join(
        [
            f"path: {dataset_path}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            names,
            "",
        ]
    )


def write_data_yaml(task):
    dataset_path = Path(task["dataset_path"])
    yaml_path = queue_dir() / f"{task['id']}.yaml"
    yaml_path.write_text(
        data_yaml_text(task, str(dataset_path)),
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
        has_best = (weights_dir / "best.pt").is_file()
        has_last = (weights_dir / "last.pt").is_file()
        if not has_best and not has_last:
            continue
        items.append(
            {
                "task": task,
                "run_dir": run_dir,
                "has_best": has_best,
                "has_last": has_last,
                "results_image": f"/model/{task.get('project', '')}/metrics/{task['id']}/files/results.png" if (run_dir / "results.png").is_file() else "",
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
            has_best = (weights_dir / "best.pt").is_file()
            has_last = (weights_dir / "last.pt").is_file()
            if not has_best and not has_last:
                continue
            items.append(
                {
                    "task": task,
                    "run_dir": resolved,
                    "has_best": has_best,
                    "has_last": has_last,
                    "results_image": f"/model/{task.get('project', '')}/metrics/{task['id']}/files/results.png" if (run_dir / "results.png").is_file() else "",
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


def train_queue_filter(queue: str):
    return queue if queue in {"all", "active", "completed"} else "all"


def task_progress(task: dict):
    if task.get("progress") is not None:
        try:
            return max(0, min(int(task.get("progress") or 0), 100))
        except (TypeError, ValueError):
            pass
    status_value = task.get("status", "")
    if status_value == COMPLETE_STATUS:
        return 100
    if status_value == "进行中":
        return 50
    if status_value in {"失败", "取消", "演示模式"}:
        return 100
    return 0


def task_card_view(task: dict):
    progress = task_progress(task)
    is_remote = task.get("train_scope") == "remote"
    resource_id = str(task.get("resource_id") or "")
    resource_url = f"/resources/{task.get('project', '')}/server/{resource_id}" if is_remote and resource_id else ""
    status_value = str(task.get("status") or "")
    dataset_distribution = task_dataset_distribution(task)
    params = []
    for key, label in (
        ("model", "模型"),
        ("epochs", "轮数"),
        ("model_version", "版本"),
        ("model_size", "尺寸"),
        ("batch", "batch"),
        ("device", "device"),
        ("workers", "workers"),
    ):
        value = task.get(key)
        if value is not None and value != "":
            params.append({"label": label, "value": value})
    for key in YOLO_OPTION_KEYS:
        if key in {"batch", "device", "workers"}:
            continue
        value = task.get(key)
        if value is not None and value != "":
            params.append({"label": key, "value": value})
    return {
        **task,
        "display_created_at": display_datetime(task.get("created_at", "")),
        "progress": progress,
        "progress_style": f"conic-gradient(#1667c7 0 {progress}%, #e2e8f0 {progress}% 100%)",
        "params": params,
        "is_active": status_value in ACTIVE_STATUSES,
        "is_completed": status_value == COMPLETE_STATUS,
        "status_class": {
            COMPLETE_STATUS: "completed",
            "失败": "failed",
            "取消": "cancelled",
            "进行中": "running",
            "排队中": "queued",
            "演示模式": "demo",
        }.get(status_value, "default"),
        "is_remote": is_remote,
        "scope_label": "远程" if is_remote else "本地",
        "resource_url": resource_url,
        "resource_name": task.get("resource_name") or "算力服务器",
        "remote_dataset_path": task.get("remote_dataset_path", ""),
        "dataset_distribution": dataset_distribution,
    }


def task_dataset_distribution(task: dict):
    dataset_path = Path(str(task.get("dataset_path") or "")).expanduser()
    splits = []
    total = 0
    for name, label, color in (
        ("train", "train", "#1667c7"),
        ("val", "val", "#16a34a"),
        ("test", "test", "#f59e0b"),
    ):
        count = count_dataset_split(dataset_path, name)["images"] if dataset_path.is_dir() else 0
        total += count
        splits.append({"name": name, "label": label, "count": count, "color": color})
    cursor = 0
    gradient = []
    for split in splits:
        percent = round(split["count"] / total * 100) if total else 0
        split["percent"] = percent
        start = cursor
        cursor += percent
        gradient.append(f"{split['color']} {start}% {cursor}%")
    if total and cursor < 100:
        gradient.append(f"{splits[-1]['color']} {cursor}% 100%")
    return {
        "total": total,
        "splits": splits,
        "chart_style": f"conic-gradient({', '.join(gradient)})" if total else "conic-gradient(#e2e8f0 0 100%)",
    }


def train_overview(tasks: list[dict]):
    total = len(tasks)
    completed = sum(1 for task in tasks if task.get("status") == COMPLETE_STATUS)
    failed = sum(1 for task in tasks if task.get("status") == "失败")
    active = sum(1 for task in tasks if task.get("status") in ACTIVE_STATUSES)
    remote = sum(1 for task in tasks if task.get("train_scope") == "remote")
    local = total - remote
    return {
        "total_count": total,
        "completed_count": completed,
        "active_count": total - completed,
        "running_count": active,
        "failed_count": failed,
        "remote_count": remote,
        "local_count": local,
        "completed_percent": round(completed / total * 100) if total else 0,
        "failed_percent": round(failed / total * 100) if total else 0,
        "remote_percent": round(remote / total * 100) if total else 0,
        "local_percent": round(local / total * 100) if total else 0,
    }


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
            image = {"name": relative, "src": f"/model/{task.get('project', '')}/metrics/{task['id']}/files/{relative}"}
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
            files.append({"name": relative, "size": path.stat().st_size, "href": f"/model/{task.get('project', '')}/metrics/{task['id']}/files/{relative}"})
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
    for key in YOLO_OPTION_KEYS:
        if key == "data":
            continue
        value = task.get(key)
        if value not in (None, ""):
            command.append(f"{key}={value}")
    return command


def remote_train_command(task, remote_yaml: str, remote_runs_root: str):
    command = [
        "yolo",
        "detect",
        "train",
        f"data={remote_yaml}",
        f"model={task['model']}",
        f"epochs={task['epochs']}",
        f"project={remote_runs_root}",
        f"name={task['name']}",
        "exist_ok=True",
    ]
    for key in YOLO_OPTION_KEYS:
        if key == "data":
            continue
        value = task.get(key)
        if value not in (None, ""):
            command.append(f"{key}={value}")
    return command


def tmux_session_name(task_id: str):
    return f"yoloutils_{task_id}"


def shell_join(command: list[str]):
    return " ".join(shlex.quote(str(part)) for part in command)


def tmux_wrap_command(command: str, log_path: str, exit_path: str, cwd: str = ""):
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    return f"{prefix}{command} >> {shlex.quote(log_path)} 2>&1; printf %s $? > {shlex.quote(exit_path)}"


def local_tmux_available():
    return shutil.which("tmux") is not None


def remote_command_output(client, command: str):
    _, stdout, stderr = client.exec_command(command, timeout=10)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), output, error


def remote_tmux_available(client):
    code, _, _ = remote_command_output(client, "command -v tmux >/dev/null 2>&1")
    return code == 0


def read_remote_text(sftp, path: str):
    try:
        with sftp.file(path, "r") as handle:
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def sftp_mkdirs(sftp, path: str):
    current = "/" if path.startswith("/") else "."
    for part in [item for item in path.split("/") if item]:
        current = posixpath.join(current, part) if current != "/" else f"/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def remote_home_dir(client, sftp):
    try:
        _, stdout, _ = client.exec_command("printf %s \"$HOME\"", timeout=10)
        home = stdout.read().decode("utf-8", errors="replace").strip()
        if home:
            return home
    except Exception:
        pass
    return sftp.normalize(".")


def expand_remote_path(path: str, remote_home: str):
    path = (path or "").strip()
    if path == "~":
        return remote_home
    if path.startswith("~/"):
        return posixpath.join(remote_home, path[2:])
    return path


def download_remote_tree(sftp, remote_path: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_path):
        if item.filename in {".", ".."}:
            continue
        remote_child = posixpath.join(remote_path, item.filename)
        local_child = local_path / item.filename
        if stat_module.S_ISDIR(item.st_mode):
            download_remote_tree(sftp, remote_child, local_child)
        else:
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))


def append_channel_output(task_id: str, channel, stderr: bool = False):
    chunks = []
    recv_ready = channel.recv_stderr_ready if stderr else channel.recv_ready
    recv = channel.recv_stderr if stderr else channel.recv
    while recv_ready():
        chunks.append(recv(4096))
    if chunks:
        append_log(task_id, b"".join(chunks).decode("utf-8", errors="replace"))


def run_remote_task(task):
    task_id = task["id"]
    run_dir = expected_run_dir(task)
    update_task(
        task_id,
        status="进行中",
        run_path=str(run_dir),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    try:
        import paramiko
    except ImportError:
        append_log(task_id, "远程训练需要 paramiko，请先安装 paramiko。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return

    resource = find_resource(workspace_path(), str(task.get("resource_id") or ""))
    if resource is None:
        append_log(task_id, "算力服务器不存在，无法启动远程训练。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    remote_dataset_path = str(task.get("remote_dataset_path") or "").strip()
    if not remote_dataset_path:
        append_log(task_id, "远程数据集路径为空，无法启动远程训练。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    try:
        append_log(task_id, f"连接算力服务器：{resource.get('name') or resource.get('host')}\n")
        client.connect(
            hostname=resource["host"],
            port=int(resource.get("port") or 22),
            username=resource.get("username") or None,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        append_log(task_id, "SSH 连接成功，打开 SFTP...\n")
        sftp = client.open_sftp()
        remote_home = remote_home_dir(client, sftp)
        append_log(task_id, f"远程 Home：{remote_home}\n")
        if not remote_tmux_available(client):
            append_log(task_id, "远程服务器未安装 tmux，无法启动持久训练。请先安装 tmux。\n")
            update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
            return
        remote_dataset_path = expand_remote_path(remote_dataset_path, remote_home)
        remote_base_dir = posixpath.join(remote_home, ".yoloutils")
        remote_work_dir = posixpath.join(remote_base_dir, "train")
        remote_runs_root = posixpath.join(remote_base_dir, "runs", task["project"])
        remote_yaml = posixpath.join(remote_work_dir, f"{task_id}.yaml")
        remote_log = posixpath.join(remote_work_dir, f"{task_id}.log")
        remote_exit = posixpath.join(remote_work_dir, f"{task_id}.exit")
        remote_run_dir = posixpath.join(remote_runs_root, task["name"])
        append_log(task_id, f"准备远程工作目录：{remote_work_dir}\n")
        sftp_mkdirs(sftp, remote_work_dir)
        append_log(task_id, f"准备远程 runs 目录：{remote_runs_root}\n")
        sftp_mkdirs(sftp, remote_runs_root)
        append_log(task_id, f"写入远程 data.yaml：{remote_yaml}\n")
        with sftp.file(remote_yaml, "w") as handle:
            handle.write(data_yaml_text(task, remote_dataset_path))
        update_task(task_id, remote_run_path=remote_run_dir)

        command = remote_train_command(task, remote_yaml, remote_runs_root)
        shell_command = shell_join(command)
        session = tmux_session_name(task_id)
        tmux_command = tmux_wrap_command(shell_command, remote_log, remote_exit, remote_work_dir)
        start_command = f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(tmux_command)}"
        append_log(task_id, f"远程数据集：{remote_dataset_path}\n")
        append_log(task_id, "$ " + start_command + "\n\n")
        code, output, error = remote_command_output(client, start_command)
        if code != 0:
            append_log(task_id, (output + error).strip() + "\n")
            append_log(task_id, f"tmux 启动失败，退出码: {code}\n")
            update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
            return
        running_processes[task_id] = {"type": "remote-tmux", "client": client, "session": session}
        remote_log_offset = 0
        try:
            while True:
                text = read_remote_text(sftp, remote_log)
                if len(text) > remote_log_offset:
                    append_log(task_id, text[remote_log_offset:])
                    remote_log_offset = len(text)
                code, _, _ = remote_command_output(client, f"tmux has-session -t {shlex.quote(session)} >/dev/null 2>&1")
                if code != 0:
                    break
                time.sleep(1)
            text = read_remote_text(sftp, remote_log)
            if len(text) > remote_log_offset:
                append_log(task_id, text[remote_log_offset:])
            exit_text = read_remote_text(sftp, remote_exit).strip()
            return_code = int(exit_text) if exit_text.isdigit() else 1
        finally:
            running_processes.pop(task_id, None)
        append_log(task_id, f"\n远程进程退出码: {return_code}\n")
        if return_code != 0:
            update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
            return

        append_log(task_id, f"下载远程 runs 目录：{remote_run_dir}\n")
        download_remote_tree(sftp, remote_run_dir, run_dir)
        append_log(task_id, f"已下载到本地：{run_dir}\n")
        update_task(task_id, status=COMPLETE_STATUS, finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as error:
        append_log(task_id, f"\n远程训练失败: {error}\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
    finally:
        if sftp is not None:
            sftp.close()
        client.close()


def run_task(task):
    if task.get("train_scope") == "remote":
        run_remote_task(task)
        return
    task_id = task["id"]
    run_dir = expected_run_dir(task)
    update_task(
        task_id,
        status="进行中",
        run_path=str(run_dir),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    command = train_command(task)
    append_log(task_id, "$ " + shell_join(command) + "\n\n")
    if shutil.which("yolo") is None:
        append_log(task_id, "yolo 命令不存在，请先安装 ultralytics 或确认虚拟环境 PATH。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    if not local_tmux_available():
        append_log(task_id, "本机未安装 tmux，无法启动持久训练。请先安装 tmux。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    session = tmux_session_name(task_id)
    exit_path = log_file(task_id).with_suffix(".exit")
    exit_path.unlink(missing_ok=True)
    tmux_command = tmux_wrap_command(shell_join(command), str(log_file(task_id)), str(exit_path), str(workspace_path()))
    start_command = ["tmux", "new-session", "-d", "-s", session, tmux_command]
    try:
        append_log(task_id, "$ " + shell_join(start_command) + "\n\n")
        started = subprocess.run(start_command, cwd=workspace_path(), text=True, capture_output=True)
        if started.returncode != 0:
            append_log(task_id, (started.stdout + started.stderr).strip() + "\n")
            append_log(task_id, f"tmux 启动失败，退出码: {started.returncode}\n")
            update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
            return
        running_processes[task_id] = {"type": "local-tmux", "session": session}
        while subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            time.sleep(1)
        try:
            return_code = int(exit_path.read_text(encoding="utf-8", errors="replace").strip())
        except (OSError, ValueError):
            return_code = 1
    except Exception as error:
        append_log(task_id, f"\n训练启动失败: {error}\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    finally:
        running_processes.pop(task_id, None)
        exit_path.unlink(missing_ok=True)

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


@router.get("/model/train")
@router.get("/train")
def train(request: Request, tab: str = "", queue: str = "all"):
    project = request.query_params.get("project", "")
    if project:
        url = f"/model/{project}/train"
        params = []
        if queue != "all":
            params.append(f"queue={queue}")
        if params:
            url += "?" + "&".join(params)
        return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    if current_project:
        tasks = [task for task in tasks if task.get("project") == current_project]
    queue_filter = train_queue_filter(queue)
    overview = train_overview(tasks)
    response = templates.TemplateResponse(
        request=request,
        name="train/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "tasks": tasks,
            "queue_tasks": [task_card_view(task) for task in filtered_queue_tasks(tasks, queue_filter)],
            "models": model_items(tasks, current_project),
            **overview,
            "queue_filter": queue_filter,
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            "current_project_name": display_project_name(workspace, current_project),
            "demo_mode": demo_mode_enabled(),
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


@router.get("/model/train/models/{task_id}")
@router.get("/train/models/{task_id}")
def train_model(request: Request, task_id: str):
    workspace = workspace_path()
    task = resolve_model_task(task_id)
    if task is None:
        return RedirectResponse(url="/model/train", status_code=status.HTTP_303_SEE_OTHER)
    current_project = task.get("project", request.cookies.get("current_project", ""))
    if request.url.path.startswith("/model/train/models/"):
        return RedirectResponse(url=f"/model/{current_project}/metrics/{task_id}", status_code=status.HTTP_303_SEE_OTHER)
    assets = run_result_assets(task)
    return templates.TemplateResponse(
        request=request,
        name="train/model.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": task,
            "assets": assets,
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )


@router.get("/model/{project}/metrics/{task_id}")
def train_model_metrics(request: Request, project: str, task_id: str):
    workspace = workspace_path()
    task = resolve_model_task(task_id)
    if task is None or task.get("project") != project:
        return RedirectResponse(url=f"/model/{project}", status_code=status.HTTP_303_SEE_OTHER)
    assets = run_result_assets(task)
    return templates.TemplateResponse(
        request=request,
        name="train/model.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": task,
            "assets": assets,
            "export_formats": MODEL_EXPORT_FORMATS,
            "active_page": "model",
            "model_active": "overview",
            "current_project": project,
            **header_context(request, workspace),
        },
    )


def zip_export_artifact(path: Path):
    temp = tempfile.NamedTemporaryFile(prefix=f"{path.name}-", suffix=".zip", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(path.rglob("*"), key=lambda child: child.relative_to(path).as_posix()):
            if item.is_file():
                archive.write(item, item.relative_to(path).as_posix())
    return temp_path


def exported_artifact(weights_dir: Path, started_at: float):
    candidates = []
    for path in weights_dir.iterdir():
        if path.name in WEIGHT_FILES:
            continue
        if not path.name.startswith("best"):
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if modified + 1 >= started_at:
            candidates.append((modified, path))
    return max(candidates, default=(0, None))[1]


@router.post("/model/{project}/metrics/{task_id}/export")
def export_model(project: str, task_id: str, request: Request):
    return JSONResponse({"ok": False, "error": "请使用表单提交导出格式"}, status_code=400)


@router.get("/model/{project}/metrics/{task_id}/export")
def export_model_download(project: str, task_id: str, format: str = "onnx"):
    export_format = (format or "onnx").strip().lower()
    if export_format not in MODEL_EXPORT_FORMATS:
        return JSONResponse({"ok": False, "error": "不支持的导出格式"}, status_code=400)
    task = resolve_model_task(task_id)
    if task is None or task.get("project") != project:
        return JSONResponse({"ok": False, "error": "模型不存在"}, status_code=404)
    run_dir = task_run_dir(task)
    weights_dir = run_dir / "weights"
    best_path = weights_dir / "best.pt"
    if not best_path.is_file():
        return JSONResponse({"ok": False, "error": "best.pt 不存在，无法导出"}, status_code=404)
    if shutil.which("yolo") is None:
        return JSONResponse({"ok": False, "error": "yolo 命令不存在，请先安装 ultralytics"}, status_code=500)
    started_at = time.time()
    result = subprocess.run(
        ["yolo", "export", f"model={best_path}", f"format={export_format}"],
        cwd=weights_dir,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return JSONResponse(
            {"ok": False, "error": result.stderr or result.stdout or "模型导出失败"},
            status_code=500,
        )
    artifact = exported_artifact(weights_dir, started_at)
    if artifact is None:
        return JSONResponse({"ok": False, "error": "导出完成，但没有找到导出文件"}, status_code=500)
    if artifact.is_dir():
        zip_path = zip_export_artifact(artifact)
        return FileResponse(
            zip_path,
            filename=f"{task['name']}-{export_format}.zip",
            background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
        )
    return FileResponse(artifact, filename=f"{task['name']}-{artifact.name}")


@router.get("/model/{project}/metrics/{task_id}/weights/{weight_name}")
@router.get("/model/train/models/{task_id}/weights/{weight_name}")
@router.get("/train/models/{task_id}/weights/{weight_name}")
def download_model_weight(task_id: str, weight_name: str, project: str = ""):
    task = resolve_model_task(task_id)
    if task is None or weight_name not in WEIGHT_FILES or (project and task.get("project") != project):
        return JSONResponse({"ok": False, "error": "模型不存在"}, status_code=404)
    run_dir = task_run_dir(task)
    path = (run_dir / "weights" / weight_name).resolve()
    weights_dir = (run_dir / "weights").resolve()
    if not is_inside(path, weights_dir) or not path.is_file():
        return JSONResponse({"ok": False, "error": "模型文件不存在"}, status_code=404)
    return FileResponse(path, filename=f"{task['name']}-{weight_name}")


@router.get("/model/{project}/metrics/{task_id}/files/{file_path:path}")
@router.get("/model/train/models/{task_id}/files/{file_path:path}")
@router.get("/train/models/{task_id}/files/{file_path:path}")
def train_model_file(task_id: str, file_path: str, project: str = ""):
    task = resolve_model_task(task_id)
    if task is None or (project and task.get("project") != project):
        return JSONResponse({"ok": False, "error": "文件不存在"}, status_code=404)
    run_dir = task_run_dir(task).resolve()
    path = (run_dir / file_path).resolve()
    if not is_inside(path, run_dir) or not path.is_file():
        return JSONResponse({"ok": False, "error": "文件不存在"}, status_code=404)
    return FileResponse(path)


@router.get("/model/train/new")
@router.get("/train/new")
def new_train(request: Request, dataset: str = ""):
    project = request.query_params.get("project", "")
    is_remote = request.query_params.get("remote") == "1"
    requested_dataset = (dataset or "").strip()
    if project:
        url = f"/model/{project}/train/new"
        params = []
        if requested_dataset:
            params.append(("dataset", requested_dataset))
        if is_remote:
            params.append(("remote", "1"))
        if params:
            url += "?" + urlencode(params)
        return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    dataset_options = remote_dataset_dirs(current_project) if is_remote else dataset_dirs(current_project)
    locked_dataset = (
        matched_remote_dataset(current_project, requested_dataset)
        if is_remote
        else matched_dataset(current_project, requested_dataset)
    )
    dataset_item = locked_dataset or (dataset_options[0] if dataset_options else None)
    response = templates.TemplateResponse(
        request=request,
        name="train/new.html",
        context={
            "request": request,
            "workspace": workspace,
            "dataset": dataset_item,
            "datasets": dataset_options,
            "dataset_locked": locked_dataset is not None,
            "remote_train": is_remote,
            "model_versions": MODEL_VERSIONS,
            "model_sizes": MODEL_SIZES,
            "default_model_version": "YOLO26",
            "default_model_size": "N",
            "demo_mode": demo_mode_enabled(),
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/train/new")
@router.get("/model/train/new/{project}")
@router.get("/train/new/{project}")
def new_train_with_project(request: Request, project: str, dataset: str = ""):
    requested_dataset = (dataset or "").strip()
    if request.url.path.startswith("/model/train/new/"):
        params = []
        if requested_dataset:
            params.append(("dataset", requested_dataset))
        if request.query_params.get("remote") == "1":
            params.append(("remote", "1"))
        suffix = "?" + urlencode(params) if params else ""
        return RedirectResponse(url=f"/model/{project}/train/new{suffix}", status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = project
    is_remote = request.query_params.get("remote") == "1"
    dataset_options = remote_dataset_dirs(project) if is_remote else dataset_dirs(project)
    locked_dataset = (
        matched_remote_dataset(project, requested_dataset)
        if is_remote
        else matched_dataset(project, requested_dataset)
    )
    dataset_item = locked_dataset or (dataset_options[0] if dataset_options else None)
    response = templates.TemplateResponse(
        request=request,
        name="train/new.html",
        context={
            "request": request,
            "workspace": workspace,
            "dataset": dataset_item,
            "datasets": dataset_options,
            "dataset_locked": locked_dataset is not None,
            "remote_train": is_remote,
            "model_versions": MODEL_VERSIONS,
            "model_sizes": MODEL_SIZES,
            "default_model_version": "YOLO26",
            "default_model_size": "N",
            "demo_mode": demo_mode_enabled(),
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.post("/model/train")
@router.post("/train")
async def create_train(request: Request):
    form = await form_fields(request)
    project = form.get("project", [""])[0]
    dataset = form.get("dataset", [""])[0]
    is_remote = form.get("train_scope", ["local"])[0] == "remote"
    dataset_item = selected_remote_dataset(project, dataset) if is_remote else selected_dataset(project, dataset)
    if dataset_item is None:
        base_url = f"/model/{project}/train/new" if project else "/model/train/new"
        suffix = "?remote=1" if is_remote else ""
        return RedirectResponse(url=f"{base_url}{suffix}", status_code=status.HTTP_303_SEE_OTHER)

    model_version = clean_model_version(form.get("model_version", ["YOLO26"])[0])
    model_size = clean_model_size(form.get("model_size", ["N"])[0])
    data_value = form.get("data", [""])[0].strip()
    yolo_options = {key: optional_value(form.get(key, [""])[0]) for key in YOLO_OPTION_KEYS}
    yolo_options["data"] = data_value
    task = {
        "id": uuid4().hex[:12],
        "name": form.get("name", ["train"])[0].strip() or "train",
        "project": dataset_item["project"],
        "dataset": dataset_item["name"],
        "dataset_path": str(dataset_item["path"]),
        "train_scope": "remote" if is_remote else "local",
        "model_version": model_version,
        "model_size": model_size,
        "model": model_weight(model_version, model_size),
        "epochs": int(form.get("epochs", ["200"])[0] or 200),
        "batch": optional_int(form.get("batch", [""])[0]),
        "device": yolo_options.get("device") or "",
        "workers": optional_int(form.get("workers", [""])[0]),
        "status": "演示模式" if demo_mode_enabled() else "排队中",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **{key: value for key, value in yolo_options.items() if key not in {"batch", "device", "workers"}},
    }
    if is_remote:
        task.update(
            {
                "resource_id": dataset_item.get("resource_id", ""),
                "resource_name": dataset_item.get("resource_name", "算力服务器"),
                "remote_dataset_path": dataset_item.get("remote_path", ""),
            }
        )
    with queue_lock:
        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
    append_log(task["id"], f"任务已创建: {task['created_at']}\n")
    if demo_mode_enabled():
        append_log(task["id"], "当前为演示模式，不会启动训练。\n")
    else:
        ensure_worker()
    return RedirectResponse(url=f"/model/{task['project']}/train", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/model/train/tasks/{task_id}")
@router.get("/train/tasks/{task_id}")
def train_task(request: Request, task_id: str):
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    tasks = load_tasks()
    task = next((item for item in tasks if item["id"] == task_id), None)
    if task is None:
        return RedirectResponse(url="/model/train", status_code=status.HTTP_303_SEE_OTHER)
    log = log_file(task_id).read_text(encoding="utf-8", errors="replace") if log_file(task_id).is_file() else ""
    return templates.TemplateResponse(
        request=request,
        name="train/task.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": task,
            "log": log,
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )


@router.get("/model/train/tasks/{task_id}/logs")
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
        "task": task_card_view(task),
        "log": log,
        "size": path.stat().st_size if path.is_file() else 0,
    }


@router.post("/model/train/tasks/{task_id}/cancel")
@router.post("/train/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    process = running_processes.get(task_id)
    if process:
        if isinstance(process, dict) and process.get("type") == "local-tmux":
            subprocess.run(["tmux", "kill-session", "-t", str(process.get("session") or "")], capture_output=True)
        elif isinstance(process, dict) and process.get("type") == "remote-tmux":
            client = process.get("client")
            session = str(process.get("session") or "")
            if client and session:
                client.exec_command(f"tmux kill-session -t {shlex.quote(session)} >/dev/null 2>&1")
        elif hasattr(process, "poll") and process.poll() is None:
            process.terminate()
        elif hasattr(process, "close"):
            process.close()
    update_task(task_id, status="取消", finished_at=datetime.now().isoformat(timespec="seconds"))
    append_log(task_id, "\n任务已取消。\n")
    return RedirectResponse(url="/model/train", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/model/train/tasks/{task_id}/delete")
@router.post("/train/tasks/{task_id}/delete")
def delete_task(task_id: str):
    with queue_lock:
        tasks = load_tasks()
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task and task.get("status") in ACTIVE_STATUSES:
            return RedirectResponse(url=f"/model/{task.get('project', '')}/train", status_code=status.HTTP_303_SEE_OTHER)
        save_tasks([item for item in tasks if item.get("id") != task_id])
    if task:
        log_file(task_id).unlink(missing_ok=True)
        return RedirectResponse(url=f"/model/{task.get('project', '')}/train", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/model/train", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/model/train/tasks/{task_id}/rerun")
@router.post("/train/tasks/{task_id}/rerun")
def rerun_task(task_id: str):
    with queue_lock:
        tasks = load_tasks()
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            return RedirectResponse(url="/model/train", status_code=status.HTTP_303_SEE_OTHER)
        if task.get("status") in ACTIVE_STATUSES:
            return RedirectResponse(url=f"/model/{task.get('project', '')}/train", status_code=status.HTTP_303_SEE_OTHER)
        task["status"] = "排队中"
        task["created_at"] = datetime.now().isoformat(timespec="seconds")
        for key in ("started_at", "finished_at", "run_path", "remote_run_path", "progress"):
            task.pop(key, None)
        save_tasks(tasks)
    log_file(task_id).unlink(missing_ok=True)
    append_log(task_id, f"任务已重跑: {task['created_at']}\n")
    ensure_worker()
    return RedirectResponse(url=f"/model/{task.get('project', '')}/train", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/model/train/{project}")
@router.get("/model/{project}/train")
@router.get("/train/{project}")
def train_with_project(request: Request, project: str, tab: str = "", queue: str = "all"):
    if request.url.path.startswith("/model/train/"):
        suffix = f"?queue={queue}" if queue != "all" else ""
        return RedirectResponse(url=f"/model/{project}/train{suffix}", status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = project
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    if current_project:
        tasks = [task for task in tasks if task.get("project") == current_project]
    queue_filter = train_queue_filter(queue)
    overview = train_overview(tasks)
    response = templates.TemplateResponse(
        request=request,
        name="train/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "tasks": tasks,
            "queue_tasks": [task_card_view(task) for task in filtered_queue_tasks(tasks, queue_filter)],
            "models": model_items(tasks, current_project),
            **overview,
            "queue_filter": queue_filter,
            "active_page": "model",
            "model_active": "train",
            "current_project": current_project,
            "current_project_name": display_project_name(workspace, current_project),
            "demo_mode": demo_mode_enabled(),
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response

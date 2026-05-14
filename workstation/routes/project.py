import json
import os
import re
import shutil
import socket
import subprocess
import time
import traceback
import getpass
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlencode

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

try:
    import psutil
except ImportError:
    psutil = None


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
PROJECT_DIR_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ANNOTATE_DIR = "annotate"
TEST_DIR = "test"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel"}
USER_HEARTBEAT_TIMEOUT = 45
PROJECT_UPLOAD_LOG = ".project.log"
WORKSPACE_LOG = ".workstation/workspace.log"
PROJECT_INDEX = ".workstation/index.json"


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def format_bytes(size: int):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{int(size)} B"


def total_memory_bytes():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return 0


def available_memory_bytes():
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            available = int(pages) * int(page_size)
            if available > 0:
                return available
        except (AttributeError, OSError, ValueError):
            pass
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    page_size_match = re.search(r"page size of (\d+) bytes", result.stdout)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    available_pages = 0
    for name in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"{re.escape(name)}:\s+(\d+)\.", result.stdout)
        if match:
            available_pages += int(match.group(1))
    if available_pages:
        return available_pages * page_size
    return 0


def capacity_chart(total: int, available: int):
    total = max(0, int(total or 0))
    available = max(0, min(int(available or 0), total))
    used = max(0, total - available)
    used_percent = round(used / total * 100, 1) if total else 0
    free_percent = round(available / total * 100, 1) if total else 0
    return {
        "total": format_bytes(total) if total else "未知",
        "available": format_bytes(available) if total else "未知",
        "used": format_bytes(used) if total else "未知",
        "used_percent": used_percent,
        "free_percent": free_percent,
        "style": f"conic-gradient(#16a34a 0 {free_percent}%, #e2e8f0 {free_percent}% 100%)" if total else "#e2e8f0",
    }


def gpu_summary():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"count": 0, "total": 0, "available": 0}
    if result.returncode != 0:
        return {"count": 0, "total": 0, "available": 0}
    totals = []
    available = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            totals.append(int(float(parts[0])))
            available.append(int(float(parts[1])))
        except ValueError:
            continue
    if not totals:
        return {"count": 0, "total": 0, "available": 0}
    return {
        "count": len(totals),
        "total": sum(totals) * 1024 * 1024,
        "available": sum(available) * 1024 * 1024,
    }


def cpu_usage_items():
    count = os.cpu_count() or 0
    if psutil is not None:
        try:
            values = psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception:
            values = []
    else:
        values = []
    if not values:
        values = [0] * count
    return [
        {"label": f"CPU {index + 1}", "percent": round(max(0, min(float(value), 100)), 1)}
        for index, value in enumerate(values)
    ]


def compute_config(workspace: Path):
    gpu = gpu_summary()
    memory = total_memory_bytes()
    cpu_items = cpu_usage_items()
    disk = shutil.disk_usage(workspace)
    return {
        "server": socket.gethostname(),
        "cpu_count": len(cpu_items) or os.cpu_count() or 0,
        "cpu_items": cpu_items,
        "memory": capacity_chart(memory, available_memory_bytes()),
        "gpu_count": gpu["count"],
        "gpu_memory": capacity_chart(gpu["total"], gpu["available"]),
        "disk": capacity_chart(disk.total, disk.free),
    }


def users_file(workspace: Path):
    return workspace / ".users"


def chat_file(workspace: Path):
    return workspace / ".team-chat.json"


def read_user_session_data(workspace: Path, prune: bool = True):
    now = time.time()
    path = users_file(workspace)
    if not path.is_file():
        return {"users": [], "projects": {}, "seen_at": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"users": [], "projects": {}, "seen_at": {}}
    users = data.get("users", [])
    projects = data.get("projects", {})
    seen_at = data.get("seen_at", {})
    users = [str(user).strip() for user in users if str(user).strip()] if isinstance(users, list) else []
    seen_at = {
        str(user).strip(): float(timestamp or 0)
        for user, timestamp in seen_at.items()
        if str(user).strip()
    } if isinstance(seen_at, dict) else {}
    if prune:
        users = [user for user in users if now - seen_at.get(user, 0) <= USER_HEARTBEAT_TIMEOUT]
    return {
        "users": users,
        "projects": {
            str(user).strip(): str(project).strip()
            for user, project in projects.items()
            if str(user).strip() in users
        } if isinstance(projects, dict) else {},
        "seen_at": {user: seen_at.get(user, now) for user in users},
    }


def read_online_users(workspace: Path):
    return read_user_session_data(workspace)["users"]


def read_user_projects(workspace: Path):
    return read_user_session_data(workspace)["projects"]


def write_online_users(workspace: Path, users: list[str]):
    session = read_user_session_data(workspace)
    projects = {
        user: project
        for user, project in session["projects"].items()
        if user in users
    }
    seen_at = {
        user: session["seen_at"].get(user, time.time())
        for user in users
    }
    users_file(workspace).write_text(
        json.dumps(
            {"users": sorted(set(users), key=str.lower), "projects": projects, "seen_at": seen_at},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_user_project(workspace: Path, username: str, project: str = ""):
    username = (username or "").strip()
    if not username:
        return
    session = read_user_session_data(workspace)
    users = sorted(set(session["users"]), key=str.lower)
    if username not in users:
        return
    projects = session["projects"]
    if project:
        projects[username] = project
    else:
        projects.pop(username, None)
    session["seen_at"][username] = time.time()
    users_file(workspace).write_text(
        json.dumps({"users": users, "projects": projects, "seen_at": session["seen_at"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def touch_user(workspace: Path, username: str):
    username = (username or "").strip()
    if not username:
        return None
    session = read_user_session_data(workspace, prune=False)
    users = sorted(set(session["users"]), key=str.lower)
    if username not in users:
        return None
    session["seen_at"][username] = time.time()
    users_file(workspace).write_text(
        json.dumps(
            {"users": users, "projects": session["projects"], "seen_at": session["seen_at"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return session


def current_username(request: Request, workspace: Path):
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    if username not in read_user_session_data(workspace, prune=False)["users"]:
        return ""
    touch_user(workspace, username)
    return username


def team_mode_enabled():
    return os.environ.get("YOLOUTILS_TEAM", "").lower() in ("1", "true", "yes", "on")


def normalize_mdns(value: str):
    name = (value or "").strip().lower()
    if "://" in name:
        name = name.split("://", 1)[1]
    name = name.split("/", 1)[0].split(":", 1)[0]
    return name if name.endswith(".local") else f"{name}.local"


def lan_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            host = item[4][0]
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass
    return ""


def share_url(request: Request):
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    mdns = os.environ.get("YOLOUTILS_MDNS", "").strip()
    host = normalize_mdns(mdns) if mdns else lan_ip_address()
    if not host:
        host = request.url.hostname or "127.0.0.1"
    return f"{request.url.scheme}://{host}:{port}"


def header_context(request: Request, workspace: Path):
    is_team_mode = team_mode_enabled()
    username = current_username(request, workspace) if is_team_mode else ""
    return {
        "username": username,
        "username_initial": username[:1],
        "username_color": user_color(username) if username else "",
        "is_team_mode": is_team_mode,
        "share_url": share_url(request) if is_team_mode else "",
    }


def user_color(value: str):
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899"]
    total = sum(ord(char) for char in value or "")
    return colors[total % len(colors)]


def user_items(users: list[str], projects: dict[str, str] | None = None, project_names: dict[str, str] | None = None):
    projects = projects or {}
    project_names = project_names or {}
    return [
        {
            "name": user,
            "initial": user[:1],
            "color": user_color(user),
            "project": projects.get(user, ""),
            "project_name": project_names.get(projects.get(user, ""), projects.get(user, "")),
        }
        for user in users
    ]


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_dir(workspace: Path, directory: str):
    path = (workspace / directory).resolve()
    if path == workspace or not is_inside(path, workspace):
        return None
    return path


def read_project_registry(workspace: Path):
    registry_file = workspace / ".project"
    if not registry_file.is_file():
        return {}
    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    projects = data.get("projects", {})
    return projects if isinstance(projects, dict) else {}


def write_project_registry(workspace: Path, projects: dict):
    payload = {"projects": projects}
    (workspace / ".project").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_project_meta(path: Path, registry: dict | None = None):
    meta_file = path / ".project"
    fallback = {"name": path.name, "directory": path.name, "description": ""}
    if registry and isinstance(registry.get(path.name), dict):
        data = registry[path.name]
        return {
            "name": str(data.get("name") or path.name),
            "directory": path.name,
            "description": str(data.get("description") or ""),
        }
    if not meta_file.is_file():
        return fallback
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    stored_directory = str(data.get("directory") or path.name)
    stale_directory = stored_directory != path.name
    return {
        "name": path.name if stale_directory else str(data.get("name") or path.name),
        "directory": path.name,
        "description": str(data.get("description") or ""),
    }


def write_project_meta(workspace: Path, path: Path, name: str, description: str):
    payload = {
        "name": name.strip() or path.name,
        "directory": path.name,
        "description": description.strip(),
    }
    registry = read_project_registry(workspace)
    registry[path.name] = payload
    write_project_registry(workspace, registry)
    (path / ".project").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def count_files(path: Path, exts: set[str]):
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def project_index_file(path: Path):
    return path / PROJECT_INDEX


def read_project_index(path: Path):
    index_file = project_index_file(path)
    if not index_file.is_file():
        return None
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_project_index(path: Path, index: dict):
    index_file = project_index_file(path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _scan_image_labels(root: Path):
    result = {
        "images": 0,
        "labels": 0,
        "txt_missing": 0,
        "txt_empty": 0,
        "txt_invalid": 0,
        "txt_valid": 0,
        "class_counts": {},
    }
    if not root.is_dir():
        return result

    classes = read_classes(root / "classes.txt")
    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        result["images"] += 1
        label_file = image_path.with_suffix(".txt")
        if not label_file.is_file():
            result["txt_missing"] += 1
            continue
        result["labels"] += 1
        try:
            lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            result["txt_invalid"] += 1
            continue
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            result["txt_empty"] += 1
            continue
        valid = True
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                valid = False
                continue
            try:
                class_id = int(parts[0])
                [float(value) for value in parts[1:]]
            except ValueError:
                valid = False
                continue
            label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
            result["class_counts"][label] = result["class_counts"].get(label, 0) + 1
        if valid:
            result["txt_valid"] += 1
        else:
            result["txt_invalid"] += 1
    return result


def build_project_index(path: Path):
    annotate = _scan_image_labels(path / ANNOTATE_DIR)
    index = {
        "version": 1,
        "annotate": annotate,
        "test": {"images": count_files(path / TEST_DIR, IMAGE_EXTS)},
        "datasets": {"count": count_dataset_dirs(path)},
        "models": {"count": count_files(path / "models", MODEL_EXTS)},
    }
    write_project_index(path, index)
    return index


def empty_project_index():
    return {
        "version": 1,
        "annotate": {
            "images": 0,
            "labels": 0,
            "txt_missing": 0,
            "txt_empty": 0,
            "txt_invalid": 0,
            "txt_valid": 0,
            "class_counts": {},
        },
        "test": {"images": 0},
        "datasets": {"count": 0},
        "models": {"count": 0},
    }


def project_index(path: Path):
    return read_project_index(path) or empty_project_index()


def count_dataset_dirs(path: Path):
    roots = [path / "datasets", path / "dataset"]
    count = 0
    for root in roots:
        if root.is_dir():
            count += sum(1 for item in root.iterdir() if item.is_dir())
    return count


def resource_chart(image_count: int, dataset_count: int, model_count: int):
    items = [
        {"key": "images", "label": "图像", "count": image_count, "color": "#2563eb"},
        {"key": "datasets", "label": "数据集", "count": dataset_count, "color": "#16a34a"},
        {"key": "models", "label": "模型", "count": model_count, "color": "#f97316"},
    ]
    total = sum(item["count"] for item in items)
    start = 0.0
    segments = []
    for item in items:
        percent = (item["count"] / total * 100) if total else 0
        end = start + percent
        item["percent"] = f"{percent:.1f}%"
        if item["count"]:
            segments.append(f"{item['color']} {start:.2f}% {end:.2f}%")
        start = end
    return {
        "total": total,
        "items": items,
        "style": f"conic-gradient({', '.join(segments)})" if segments else "#e2e8f0",
    }


def read_classes(path: Path):
    if not path.is_file():
        return []
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def project_dashboard(projects: list[dict]):
    total_images = 0
    total_labels = 0
    total_test_images = 0
    total_models = 0
    total_classes_files = 0
    class_counts: dict[str, int] = {}
    colors = ["#2563eb", "#16a34a", "#f97316", "#7c3aed"]

    for project in projects:
        path = project["path"]
        index = project_index(path)
        annotate = index.get("annotate", {})
        total_images += int(annotate.get("images") or 0)
        total_labels += int(annotate.get("labels") or 0)
        total_test_images += int(index.get("test", {}).get("images") or 0)
        total_models += int(index.get("models", {}).get("count") or 0)
        total_classes_files += 1 if (path / ANNOTATE_DIR / "classes.txt").is_file() or (path / "classes.txt").is_file() else 0
        for label, count in (annotate.get("class_counts") or {}).items():
            class_counts[label] = class_counts.get(label, 0) + int(count or 0)

    resource_items = [
        {"label": "标注资源", "count": total_images + total_labels, "detail": f"{total_images} 图像 / {total_labels} txt", "color": colors[0]},
        {"label": "测试资源", "count": total_test_images, "detail": f"{total_test_images} 图像", "color": colors[1]},
        {"label": "模型资源", "count": total_models, "detail": f"{total_models} 模型", "color": colors[2]},
    ]
    resource_total = sum(item["count"] for item in resource_items)
    resource_start = 0.0
    resource_segments = []
    for item in resource_items:
        percent = (item["count"] / resource_total * 100) if resource_total else 0
        resource_end = resource_start + percent
        item["percent"] = f"{percent:.1f}%"
        if item["count"]:
            resource_segments.append(f"{item['color']} {resource_start:.2f}% {resource_end:.2f}%")
        resource_start = resource_end

    annotate_items = [
        {"label": "图像数量", "count": total_images, "detail": f"{total_images} 图像", "color": "#2563eb"},
        {"label": ".txt 数量", "count": total_labels, "detail": f"{total_labels} txt", "color": "#16a34a"},
        {"label": "classes.txt", "count": total_classes_files, "detail": f"{total_classes_files} 文件", "color": "#7c3aed"},
    ]
    annotate_total = sum(item["count"] for item in annotate_items)
    annotate_start = 0.0
    annotate_segments = []
    for item in annotate_items:
        percent = (item["count"] / annotate_total * 100) if annotate_total else 0
        annotate_end = annotate_start + percent
        item["percent"] = f"{percent:.1f}%"
        if item["count"]:
            annotate_segments.append(f"{item['color']} {annotate_start:.2f}% {annotate_end:.2f}%")
        annotate_start = annotate_end

    total_annotations = sum(class_counts.values())
    legend = []
    start = 0.0
    segments = []
    for index, (label, count) in enumerate(sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))):
        percent = (count / total_annotations * 100) if total_annotations else 0
        end = start + percent
        color = colors[index % len(colors)]
        segments.append(f"{color} {start:.2f}% {end:.2f}%")
        legend.append(
            {
                "label": label,
                "count": count,
                "percent": f"{percent:.1f}%",
                "color": color,
            }
        )
        start = end

    return {
        "image_count": total_images,
        "label_count": total_labels,
        "remaining_count": max(total_images - total_labels, 0),
        "progress_percent": max(0, min(round((total_labels / total_images * 100) if total_images else 0), 100)),
        "test_count": total_test_images,
        "model_count": total_models,
        "classes_count": total_classes_files,
        "resource_items": resource_items,
        "resource_chart_style": f"conic-gradient({', '.join(resource_segments)})" if resource_segments else "#e2e8f0",
        "annotate_items": annotate_items,
        "annotate_chart_style": f"conic-gradient({', '.join(annotate_segments)})" if annotate_segments else "#e2e8f0",
        "total_annotations": total_annotations,
        "legend": legend,
        "chart_style": f"conic-gradient({', '.join(segments)})" if segments else "#e2e8f0",
    }


def project_items(workspace: Path):
    projects = []
    if not workspace.is_dir():
        return projects

    registry = read_project_registry(workspace)
    for path in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        children = {child.name for child in path.iterdir() if child.is_dir()}
        meta = read_project_meta(path, registry)
        index = project_index(path)
        image_count = int(index.get("annotate", {}).get("images") or 0)
        test_count = int(index.get("test", {}).get("images") or 0)
        dataset_count = int(index.get("datasets", {}).get("count") or 0)
        model_count = int(index.get("models", {}).get("count") or 0)
        projects.append(
            {
                **meta,
                "path": path,
                "images": ANNOTATE_DIR in children,
                "dataset": "datasets" in children or "dataset" in children,
                "models": "models" in children,
                "image_count": image_count,
                "test_count": test_count,
                "dataset_count": dataset_count,
                "model_count": model_count,
                "resource_chart": resource_chart(image_count, dataset_count, model_count),
            }
        )
    return projects


def workspace_log_file(workspace: Path):
    return workspace / WORKSPACE_LOG


def write_error_log(workspace: Path, source: str):
    log_file = workspace_log_file(workspace)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {source}\n{traceback.format_exc()}\n"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / WORKSPACE_LOG
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("a", encoding="utf-8") as handle:
            handle.write(entry)


def read_team_chat(workspace: Path, limit: int = 200):
    path = chat_file(workspace)
    if not path.is_file():
        return []
    try:
        messages = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(messages, list):
        return []
    return messages[-limit:]


def append_team_chat(workspace: Path, username: str, message: str):
    message = (message or "").strip()
    if not message:
        return None
    messages = read_team_chat(workspace, limit=500)
    item = {
        "id": f"{int(time.time() * 1000)}-{len(messages)}",
        "username": username,
        "initial": username[:1],
        "color": user_color(username),
        "message": message[:1000],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    messages.append(item)
    chat_file(workspace).write_text(
        json.dumps(messages[-500:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


def upload_log_file(path: Path):
    return path / PROJECT_UPLOAD_LOG


def append_upload_log(path: Path, action: str, entries: list[str] | None = None):
    entries = entries or []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = upload_log_file(path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    detail = f" | {', '.join(entries)}" if entries else ""
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {action}{detail}\n")


def upload_log_lines(path: Path, limit: int = 200):
    log_file = upload_log_file(path)
    if not log_file.is_file():
        return []
    try:
        return log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def relative_log_entry(path: Path, target: Path):
    try:
        display = target.relative_to(path)
    except ValueError:
        display = target
    size = target.stat().st_size if target.is_file() else 0
    return f"{display} ({size} bytes)"


def validate_project(directory: str, name: str):
    directory = (directory or "").strip()
    name = (name or "").strip()
    if not name:
        return None, None, "项目名不能为空"
    if not directory:
        return None, None, "目录名不能为空"
    if "/" in directory or "\\" in directory or directory in (".", ".."):
        return None, None, "目录名不能包含路径分隔符"
    if not PROJECT_DIR_PATTERN.match(directory):
        return None, None, "目录名只能包含字母、数字、点、下划线和连字符"
    return directory, name, None


def validate_project_update(name: str):
    name = (name or "").strip()
    if not name:
        return None, "项目名不能为空"
    return name, None


def project_redirect(error: str = None):
    url = "/project"
    if error:
        url = f"{url}?{urlencode({'error': error})}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def refresh_project_resources(workspace: Path):
    refreshed = 0
    if not workspace.is_dir():
        return refreshed
    for path in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        build_project_index(path)
        refreshed += 1
    return refreshed


def project_detail_redirect(directory: str, error: str = None):
    url = f"/project/{directory}"
    if error:
        url = f"{url}?{urlencode({'error': error})}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def login_redirect(error: str = None):
    url = "/login"
    if error:
        url = f"{url}?{urlencode({'error': error})}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def require_team_login(request: Request, workspace: Path):
    if not team_mode_enabled():
        return None
    return None if current_username(request, workspace) else login_redirect()


async def form_fields(request: Request):
    body = (await request.body()).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


def ensure_project_structure(path: Path):
    for subdir in (ANNOTATE_DIR, TEST_DIR, "datasets", "models"):
        (path / subdir).mkdir(parents=True, exist_ok=True)


def upload_relative_path(filename: str):
    filename = (filename or "").replace("\\", "/").strip("/")
    if not filename:
        return None
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return Path(*path.parts)


def save_upload(filename: str, content: bytes, target_dir: Path):
    relative = upload_relative_path(filename)
    if relative is None:
        return None
    target = (target_dir / relative).resolve()
    target_root = target_dir.resolve()
    if not is_inside(target, target_root):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


async def uploaded_files(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return []

    body = await request.body()
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    )
    files = []
    for part in message.iter_parts():
        filename = part.get_filename()
        if not filename:
            continue
        payload = part.get_payload(decode=True) or b""
        files.append((filename, payload))
    return files


@router.get("/project")
def project(request: Request):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    is_team_mode = team_mode_enabled()
    username = current_username(request, workspace) if is_team_mode else ""
    online_users = read_online_users(workspace) if is_team_mode else []
    projects = project_items(workspace)
    project_names = {project["directory"]: project["name"] for project in projects}
    try:
        response = templates.TemplateResponse(
            request=request,
            name="project/index.html",
            context={
                "request": request,
                "workspace": workspace,
                "projects": projects,
                "error": request.query_params.get("error"),
                "active_page": "project",
                "show_create_project": True,
                "online_users": user_items(online_users, read_user_projects(workspace), project_names),
                **header_context(request, workspace),
            },
        )
        response.delete_cookie("current_project")
        return response
    except Exception:
        write_error_log(workspace, "project page error")
        return PlainTextResponse(
            f"Project page error. See {WORKSPACE_LOG}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/project/refresh")
def refresh_projects(request: Request):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    try:
        refresh_project_resources(workspace)
    except Exception:
        write_error_log(workspace, "project refresh error")
        return project_redirect("刷新项目资源失败")
    return project_redirect()


@router.get("/login")
def login_page(request: Request):
    workspace = workspace_path()
    if team_mode_enabled() and current_username(request, workspace):
        return RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
    projects = project_items(workspace)
    project_names = {project["directory"]: project["name"] for project in projects}
    online_users = read_online_users(workspace)
    return templates.TemplateResponse(
        request=request,
        name="team/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "error": request.query_params.get("error"),
            "active_page": "team",
            **header_context(request, workspace),
            "username": "",
            "username_initial": "",
            "username_color": "",
            "online_users": user_items(online_users, read_user_projects(workspace), project_names),
            "chat_messages": [],
            "current_project": "",
            "login_mode": True,
        },
    )


@router.get("/team")
def team(request: Request):
    workspace = workspace_path()
    username = current_username(request, workspace)
    if team_mode_enabled() and not username:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    requested_project = request.query_params.get("project") or request.cookies.get("current_project", "")
    if requested_project and project_dir(workspace, requested_project):
        return RedirectResponse(url=f"/team/{requested_project}", status_code=status.HTTP_303_SEE_OTHER)
    current_project = ""
    projects = project_items(workspace)
    project_names = {project["directory"]: project["name"] for project in projects}
    online_users = read_online_users(workspace)
    response = templates.TemplateResponse(
        request=request,
        name="team/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "error": request.query_params.get("error"),
            "active_page": "team",
            **header_context(request, workspace),
            "username": username,
            "username_initial": username[:1],
            "username_color": user_color(username) if username else "",
            "online_users": user_items(online_users, read_user_projects(workspace), project_names),
            "chat_messages": read_team_chat(workspace),
            "current_project": current_project,
        },
    )
    return response


@router.get("/team/{directory}")
def team_with_project(directory: str, request: Request):
    workspace = workspace_path()
    username = current_username(request, workspace)
    if team_mode_enabled() and not username:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    current_project = directory if project_dir(workspace, directory) else ""
    projects = project_items(workspace)
    project_names = {project["directory"]: project["name"] for project in projects}
    online_users = read_online_users(workspace)
    response = templates.TemplateResponse(
        request=request,
        name="team/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "error": request.query_params.get("error"),
            "active_page": "team",
            **header_context(request, workspace),
            "username": username,
            "username_initial": username[:1],
            "username_color": user_color(username) if username else "",
            "online_users": user_items(online_users, read_user_projects(workspace), project_names),
            "chat_messages": read_team_chat(workspace),
            "current_project": current_project,
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.post("/team/login")
async def login(request: Request):
    form = await form_fields(request)
    workspace = workspace_path()
    username = (form.get("username", [""])[0] or "").strip()
    users = read_online_users(workspace)
    if not username:
        return login_redirect("请输入用户名")
    if username in users:
        return login_redirect("用户名已存在，请更换用户名")
    users.append(username)
    write_online_users(workspace, users)
    response = RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie("workstation_username", quote(username), httponly=True, samesite="lax")
    return response


@router.post("/project/login")
async def legacy_login(request: Request):
    return await login(request)


@router.post("/team/logout")
def logout(request: Request):
    workspace = workspace_path()
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    users = [user for user in read_online_users(workspace) if user != username]
    write_online_users(workspace, users)
    response = RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("workstation_username")
    response.delete_cookie("current_project")
    return response


@router.post("/project/logout")
def legacy_logout(request: Request):
    return logout(request)


@router.post("/team/heartbeat")
def heartbeat(request: Request):
    workspace = workspace_path()
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    session = touch_user(workspace, username)
    if session is None:
        return JSONResponse({"ok": False, "error": "登录已过期", "users": []}, status_code=401)
    projects = project_items(workspace)
    project_names = {project["directory"]: project["name"] for project in projects}
    return {
        "ok": True,
        "users": user_items(read_online_users(workspace), read_user_projects(workspace), project_names),
    }


@router.get("/team/chat")
def team_chat(request: Request):
    workspace = workspace_path()
    username = current_username(request, workspace)
    if not username:
        return JSONResponse({"ok": False, "error": "请先登录团队"}, status_code=401)
    return {"ok": True, "messages": read_team_chat(workspace)}


@router.post("/team/chat")
async def send_team_chat(request: Request):
    workspace = workspace_path()
    username = current_username(request, workspace)
    if not username:
        return JSONResponse({"ok": False, "error": "请先登录团队"}, status_code=401)
    payload = await request.json()
    item = append_team_chat(workspace, username, str(payload.get("message", "")))
    if item is None:
        return JSONResponse({"ok": False, "error": "消息不能为空"}, status_code=400)
    return {"ok": True, "message": item, "messages": read_team_chat(workspace)}


@router.post("/project/heartbeat")
def legacy_heartbeat(request: Request):
    return heartbeat(request)


@router.post("/project")
async def create_project(request: Request):
    form = await form_fields(request)
    workspace = workspace_path()
    directory, name, error = validate_project(
        form.get("directory", [""])[0],
        form.get("name", [""])[0],
    )
    description = form.get("description", [""])[0]
    if error:
        return project_redirect(error)

    path = project_dir(workspace, directory)
    if path is None:
        error = "项目目录必须位于 workspace 内"
    elif path.exists():
        error = "项目已存在"
    if error:
        return project_redirect(error)

    ensure_project_structure(path)
    write_project_meta(workspace, path, name, description)
    return RedirectResponse(url=f"/project/{directory}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/project/{directory}/edit")
async def edit_project(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return project_redirect("项目不存在")

    form = await form_fields(request)
    name, error = validate_project_update(form.get("name", [""])[0])
    if error:
        return project_redirect(error)

    description = form.get("description", [""])[0]
    write_project_meta(workspace, path, name, description)
    return project_redirect()


@router.post("/project/{directory}/delete")
def delete_project(directory: str):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return project_redirect("项目不存在")
    shutil.rmtree(path)
    registry = read_project_registry(workspace)
    if directory in registry:
        del registry[directory]
        write_project_registry(workspace, registry)
    return project_redirect()


@router.post("/project/{directory}/refresh")
def refresh_project_detail(directory: str, request: Request):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return project_redirect("项目不存在")
    try:
        build_project_index(path)
        append_upload_log(path, "刷新项目资源索引")
    except Exception:
        write_error_log(workspace, "project detail refresh error")
        return project_detail_redirect(directory, "刷新项目资源失败")
    return project_detail_redirect(directory)


@router.get("/project/{directory}")
def project_detail(directory: str, request: Request):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    is_team_mode = team_mode_enabled()
    username = current_username(request, workspace) if is_team_mode else ""
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return project_redirect("项目不存在")

    ensure_project_structure(path)
    if is_team_mode:
        write_user_project(workspace, username, directory)
    try:
        meta = read_project_meta(path, read_project_registry(workspace))
        index = project_index(path)
        image_count = int(index.get("annotate", {}).get("images") or 0)
        test_count = int(index.get("test", {}).get("images") or 0)
        model_count = int(index.get("models", {}).get("count") or 0)
        dashboard = project_dashboard([{"path": path}])
        has_classes = (path / ANNOTATE_DIR / "classes.txt").is_file()
        project_ready = image_count > 0
        projects_by_user = read_user_projects(workspace)
        project_users = [
            user
            for user in user_items(read_online_users(workspace), projects_by_user, {directory: meta["name"]})
            if user["project"] == directory
        ] if is_team_mode else []
        response = templates.TemplateResponse(
            request=request,
            name="project/detail.html",
            context={
                "request": request,
                "workspace": workspace,
                "project": {
                    **meta,
                    "path": path,
                    "image_count": image_count,
                    "test_count": test_count,
                    "model_count": model_count,
                    "has_images": image_count > 0,
                    "has_models": model_count > 0,
                    "has_classes": has_classes,
                    "project_ready": project_ready,
                    "classes_text": (path / ANNOTATE_DIR / "classes.txt").read_text(encoding="utf-8") if has_classes else "",
                },
                "remote_user": getpass.getuser(),
                "dashboard": dashboard,
                "error": request.query_params.get("error"),
                "active_page": "project",
                "show_create_project": False,
                "current_project": directory,
                "is_team_mode": is_team_mode,
                "project_users": project_users,
                "footer_console_url": f"/project/{directory}/logs",
                "project_ready": project_ready,
                **header_context(request, workspace),
            },
        )
        response.set_cookie("current_project", directory, httponly=True, samesite="lax")
        return response
    except Exception:
        write_error_log(workspace, "project detail error")
        return PlainTextResponse(
            f"Project detail error. See {WORKSPACE_LOG}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/project/{directory}/logs")
def project_logs(directory: str):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    log_file = upload_log_file(path)
    return {
        "ok": True,
        "file": str(log_file),
        "lines": upload_log_lines(path),
    }


@router.post("/project/{directory}/upload/images")
async def upload_images(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    files = await uploaded_files(request)
    saved = [save_upload(filename, content, path / ANNOTATE_DIR) for filename, content in files]
    saved = [item for item in saved if item is not None]
    append_upload_log(
        path,
        f"上传图片/文件：接收 {len(files)} 个，保存 {len(saved)} 个",
        [relative_log_entry(path, item) for item in saved],
    )
    index = build_project_index(path)
    return {"ok": True, "saved": len(saved), "count": int(index.get("annotate", {}).get("images") or 0)}


@router.post("/project/{directory}/upload/test")
async def upload_test_images(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    files = await uploaded_files(request)
    saved = [save_upload(filename, content, path / TEST_DIR) for filename, content in files]
    saved = [item for item in saved if item is not None]
    append_upload_log(
        path,
        f"上传测试图片/文件：接收 {len(files)} 个，保存 {len(saved)} 个",
        [relative_log_entry(path, item) for item in saved],
    )
    index = build_project_index(path)
    return {"ok": True, "saved": len(saved), "count": int(index.get("test", {}).get("images") or 0)}


@router.post("/project/{directory}/upload/classes")
async def upload_classes(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)

    files = await uploaded_files(request)
    for filename, content in files:
        if PurePosixPath((filename or "").replace("\\", "/")).name.lower() != "classes.txt":
            continue
        target = path / ANNOTATE_DIR / "classes.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        append_upload_log(path, "上传 classes.txt", [relative_log_entry(path, target)])
        build_project_index(path)
        return {"ok": True, "saved": 1}
    return JSONResponse({"ok": False, "error": "请选择 classes.txt"}, status_code=400)


@router.post("/project/{directory}/classes")
async def save_classes(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)

    payload = await request.json()
    content = str(payload.get("content", "")).strip()
    if not content:
        return JSONResponse({"ok": False, "error": "classes.txt 不能为空"}, status_code=400)
    target = path / ANNOTATE_DIR / "classes.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content + "\n", encoding="utf-8")
    class_count = len([line for line in content.splitlines() if line.strip()])
    append_upload_log(path, f"保存 classes.txt：{class_count} 个标签", [relative_log_entry(path, target)])
    build_project_index(path)
    return {"ok": True}


@router.post("/project/{directory}/upload/model")
async def upload_model(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    files = await uploaded_files(request)
    saved = [save_upload(filename, content, path / "models") for filename, content in files]
    saved = [item for item in saved if item is not None]
    append_upload_log(
        path,
        f"上传模型：接收 {len(files)} 个，保存 {len(saved)} 个",
        [relative_log_entry(path, item) for item in saved],
    )
    index = build_project_index(path)
    return {"ok": True, "saved": len(saved), "count": int(index.get("models", {}).get("count") or 0)}

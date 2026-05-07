import json
import os
import re
import shutil
import socket
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


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
PROJECT_DIR_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel"}
USER_HEARTBEAT_TIMEOUT = 45
PROJECT_UPLOAD_LOG = ".yoloutils-upload.log"


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


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
    return {
        "name": str(data.get("name") or path.name),
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
        projects.append(
            {
                **meta,
                "path": path,
                "images": "images" in children,
                "dataset": "datasets" in children or "dataset" in children,
                "models": "models" in children,
                "image_count": count_files(path / "images", IMAGE_EXTS),
                "model_count": count_files(path / "models", MODEL_EXTS),
            }
        )
    return projects


def write_error_log(workspace: Path, filename: str):
    log_file = workspace / filename
    try:
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / filename
        fallback.write_text(traceback.format_exc(), encoding="utf-8")


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
    for subdir in ("images", "datasets", "models"):
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
        write_error_log(workspace, ".yoloutils-project-error.log")
        return PlainTextResponse(
            "Project page error. See .yoloutils-project-error.log",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


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
    current_project = requested_project if project_dir(workspace, requested_project) else ""
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
        image_count = count_files(path / "images", IMAGE_EXTS)
        model_count = count_files(path / "models", MODEL_EXTS)
        has_classes = (path / "images" / "classes.txt").is_file()
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
                    "model_count": model_count,
                    "has_images": image_count > 0,
                    "has_models": model_count > 0,
                    "has_classes": has_classes,
                    "project_ready": project_ready,
                    "classes_text": (path / "images" / "classes.txt").read_text(encoding="utf-8") if has_classes else "",
                },
                "remote_user": getpass.getuser(),
                "error": request.query_params.get("error"),
                "active_page": "project",
                "show_create_project": False,
                "current_project": directory,
                "project_users": project_users,
                "footer_console_url": f"/project/{directory}/logs",
                "project_ready": project_ready,
                **header_context(request, workspace),
            },
        )
        response.set_cookie("current_project", directory, httponly=True, samesite="lax")
        return response
    except Exception:
        write_error_log(workspace, ".yoloutils-project-detail-error.log")
        return PlainTextResponse(
            "Project detail error. See .yoloutils-project-detail-error.log",
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
    saved = [save_upload(filename, content, path / "images") for filename, content in files]
    saved = [item for item in saved if item is not None]
    append_upload_log(
        path,
        f"上传图片/文件：接收 {len(files)} 个，保存 {len(saved)} 个",
        [relative_log_entry(path, item) for item in saved],
    )
    return {"ok": True, "saved": len(saved), "count": count_files(path / "images", IMAGE_EXTS)}


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
        target = path / "images" / "classes.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        append_upload_log(path, "上传 classes.txt", [relative_log_entry(path, target)])
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
    target = path / "images" / "classes.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content + "\n", encoding="utf-8")
    class_count = len([line for line in content.splitlines() if line.strip()])
    append_upload_log(path, f"保存 classes.txt：{class_count} 个标签", [relative_log_entry(path, target)])
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
    return {"ok": True, "saved": len(saved), "count": count_files(path / "models", MODEL_EXTS)}

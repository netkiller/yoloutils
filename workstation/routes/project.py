import json
import os
import re
import shutil
import traceback
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


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def users_file(workspace: Path):
    return workspace / ".users"


def read_user_session_data(workspace: Path):
    path = users_file(workspace)
    if not path.is_file():
        return {"users": [], "projects": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"users": [], "projects": {}}
    users = data.get("users", [])
    projects = data.get("projects", {})
    return {
        "users": [str(user).strip() for user in users if str(user).strip()] if isinstance(users, list) else [],
        "projects": {
            str(user).strip(): str(project).strip()
            for user, project in projects.items()
            if str(user).strip()
        } if isinstance(projects, dict) else {},
    }


def read_online_users(workspace: Path):
    return read_user_session_data(workspace)["users"]


def read_user_projects(workspace: Path):
    return read_user_session_data(workspace)["projects"]


def write_online_users(workspace: Path, users: list[str]):
    projects = {
        user: project
        for user, project in read_user_projects(workspace).items()
        if user in users
    }
    users_file(workspace).write_text(
        json.dumps(
            {"users": sorted(set(users), key=str.lower), "projects": projects},
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
    users_file(workspace).write_text(
        json.dumps({"users": users, "projects": projects}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def current_username(request: Request, workspace: Path):
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    return username if username in read_online_users(workspace) else ""


def team_mode_enabled():
    return os.environ.get("YOLOUTILS_TEAM", "").lower() in ("1", "true", "yes", "on")


def header_context(request: Request, workspace: Path):
    username = current_username(request, workspace)
    is_team_mode = team_mode_enabled()
    return {
        "username": username,
        "username_initial": username[:1],
        "username_color": user_color(username) if username else "",
        "is_team_mode": is_team_mode,
        "edition_label": "企业版" if is_team_mode else "社区版",
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
                "dataset": "dataset" in children,
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


async def form_fields(request: Request):
    body = (await request.body()).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


def ensure_project_structure(path: Path):
    for subdir in ("images", "dataset", "models"):
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
    username = current_username(request, workspace)
    online_users = read_online_users(workspace)
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
                "login_required": not username,
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


@router.post("/project/login")
async def login(request: Request):
    form = await form_fields(request)
    workspace = workspace_path()
    username = (form.get("username", [""])[0] or "").strip()
    users = read_online_users(workspace)
    if not username:
        return project_redirect("请输入用户名")
    if username in users:
        return project_redirect("用户名已存在，请更换用户名")
    users.append(username)
    write_online_users(workspace, users)
    response = RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie("workstation_username", quote(username), httponly=True, samesite="lax")
    return response


@router.post("/project/logout")
def logout(request: Request):
    workspace = workspace_path()
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    users = [user for user in read_online_users(workspace) if user != username]
    write_online_users(workspace, users)
    response = RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("workstation_username")
    response.delete_cookie("current_project")
    return response


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
    username = current_username(request, workspace)
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return project_redirect("项目不存在")

    ensure_project_structure(path)
    write_user_project(workspace, username, directory)
    try:
        meta = read_project_meta(path, read_project_registry(workspace))
        image_count = count_files(path / "images", IMAGE_EXTS)
        model_count = count_files(path / "models", MODEL_EXTS)
        has_classes = (path / "images" / "classes.txt").is_file()
        project_ready = image_count > 0
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
                "error": request.query_params.get("error"),
                "active_page": "project",
                "show_create_project": False,
                "current_project": directory,
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


@router.post("/project/{directory}/upload/images")
async def upload_images(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    files = await uploaded_files(request)
    saved = [save_upload(filename, content, path / "images") for filename, content in files]
    saved = [item for item in saved if item is not None]
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
    return {"ok": True, "saved": len(saved), "count": count_files(path / "models", MODEL_EXTS)}

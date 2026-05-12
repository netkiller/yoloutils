import os
import asyncio
import json
import html
import time
import traceback
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

from workstation import Workstation


templates = Jinja2Templates(directory=TEMPLATES_DIR)
USER_HEARTBEAT_TIMEOUT = 45


class SiteWorkstation(Workstation):
    def _directory_tree(self, path: Path, include_children: bool = True):
        tree = super()._directory_tree(path, include_children=include_children)
        if path == self.workspace and path.name == "annotate":
            tree["name"] = getattr(self, "root_label", "") or "根目录"
        return tree


def site_workspace():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else PROJECT_ROOT


def read_users_data():
    path = site_workspace() / ".users"
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
    projects = {
        str(user).strip(): str(directory).strip()
        for user, directory in projects.items()
        if str(user).strip()
    } if isinstance(projects, dict) else {}
    seen_at = {
        str(user).strip(): float(timestamp or 0)
        for user, timestamp in seen_at.items()
        if str(user).strip()
    } if isinstance(seen_at, dict) else {}
    return {"users": users, "projects": projects, "seen_at": seen_at}


def read_online_users():
    now = time.time()
    data = read_users_data()
    users = data["users"]
    seen_at = data["seen_at"]
    return [user for user in users if now - seen_at.get(user, 0) <= USER_HEARTBEAT_TIMEOUT]


def write_user_project(username: str, project: str):
    username = (username or "").strip()
    if not username:
        return
    path = site_workspace() / ".users"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    users = data.get("users", [])
    users = [str(user).strip() for user in users if str(user).strip()] if isinstance(users, list) else []
    if username not in users:
        return
    projects = data.get("projects", {})
    projects = {
        str(user).strip(): str(directory).strip()
        for user, directory in projects.items()
        if str(user).strip()
    } if isinstance(projects, dict) else {}
    if project:
        projects[username] = project
    else:
        projects.pop(username, None)
    seen_at = data.get("seen_at", {})
    seen_at = {
        str(user).strip(): float(timestamp or 0)
        for user, timestamp in seen_at.items()
        if str(user).strip()
    } if isinstance(seen_at, dict) else {}
    seen_at[username] = time.time()
    path.write_text(
        json.dumps(
            {"users": sorted(set(users), key=str.lower), "projects": projects, "seen_at": seen_at},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def current_username(request: Request):
    username = unquote(request.cookies.get("workstation_username") or "").strip()
    if username not in read_users_data()["users"]:
        return ""
    write_user_project(username, request.cookies.get("current_project", ""))
    return username


def user_color(value: str):
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899"]
    total = sum(ord(char) for char in value or "")
    return colors[total % len(colors)]


def user_items(users: list[str]):
    return [
        {
            "name": user,
            "initial": user[:1],
            "color": user_color(user),
        }
        for user in users
    ]


def team_mode_enabled():
    return os.environ.get("YOLOUTILS_TEAM", "").lower() in ("1", "true", "yes", "on")


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
    images_dir = project_dir / "annotate"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir.resolve()


def project_root_workspace(project: str):
    if not project:
        return None
    workspace = site_workspace()
    project_dir = (workspace / project).resolve()
    if project_dir == workspace or not is_inside(project_dir, workspace):
        return None
    return project_dir if project_dir.is_dir() else None


def project_display_name(project: str):
    if not project:
        return "根目录"
    workspace = site_workspace()
    project_dir = (workspace / project).resolve()
    if project_dir == workspace or not is_inside(project_dir, workspace):
        return project
    meta_file = project_dir / ".project"
    if not meta_file.is_file():
        return project_dir.name
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return str(data.get("name") or project_dir.name)
    except (OSError, json.JSONDecodeError):
        return project_dir.name


def write_error_log(workstation: Workstation, error: Exception):
    workspace = workstation.workspace if workstation.workspace else PROJECT_ROOT
    log_file = workspace / ".yoloutils-annotate-error.log"
    try:
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        fallback = PROJECT_ROOT / ".yoloutils-annotate-error.log"
        fallback.write_text(traceback.format_exc(), encoding="utf-8")


def workstation_html(workstation: Workstation, active_mode: str = "annotate", project: str = "", username: str = ""):
    html = workstation._html()
    escaped_username = html_escape(username)
    online_users = user_items(read_online_users())
    project_url = f"/project/{quote(project, safe='')}" if project else "/project"
    project_query = f"?project={quote(project, safe='')}" if project else ""
    team_url = f"/team/{quote(project, safe='')}" if project else "/team"
    close_project_button = (
        '<button class="enterprise-link close-current-project" type="button" title="关闭当前项目" onclick="location.href=\'/project\'">'
        '<span class="header-icon">×</span><span>关闭项目</span></button>'
        if project else ""
    )
    project_button = (
        '<button id="projectButton" class="header-button" '
        f'title="项目" onclick="location.href=\'{project_url}\'">'
        '<span class="header-icon">▤</span><span>项目</span></button>'
    )
    resources_url = f"/resources/{quote(project, safe='')}" if project else "/resources"
    resources_button = (
        '<button id="resourcesButton" class="header-button" title="算力" '
        f'onclick="location.href=\'{resources_url}\'">'
        '<span class="header-icon">▥</span><span>算力</span></button>'
    )
    team_button = (
        '<button id="teamButton" class="header-button" title="团队" '
        f'onclick="location.href=\'{team_url}\'">'
        '<span class="header-icon">◌</span><span>团队</span></button>'
        if team_mode_enabled()
        else ""
    )
    user_header = (
        ""
        f'<span class="enterprise-link user-avatar-link" style="background:{user_color(username)}">'
        f'{html_escape(username[:1])}</span>'
        f'<span class="enterprise-link username-link">{escaped_username}</span>'
        '<form method="post" action="/team/logout" style="margin:0"><button class="enterprise-link" type="submit">注销</button></form>'
        if team_mode_enabled()
        else ""
    )
    html = (
        html.replace('"/api/', '"/annotate/api/')
        .replace("'/api/", "'/annotate/api/")
        .replace("`/api/", "`/annotate/api/")
        .replace('"/media', '"/annotate/media')
        .replace("'/media", "'/annotate/media")
        .replace("`/media", "`/annotate/media")
        .replace(
            '<a class="brand-link" href="https://www.netkiller.cn" target="_blank" rel="noopener noreferrer">Yolo Workstation</a>',
            f"{user_header}{close_project_button}",
            1,
        )
        .replace(
            "main { height: calc(100vh - 88px);",
            "main { height: calc(100vh - 84px);",
            1,
        )
        .replace(
            "body.console-open main { height: calc(100vh - 88px - var(--console-height, 160px) - 4px);",
            "body.console-open main { height: calc(100vh - 84px - var(--console-height, 160px) - 4px);",
            1,
        )
        .replace(
            '<button id="annotateModeButton"',
            f'{team_button}{project_button}{resources_button}<button id="annotateModeButton"',
            1,
        )
        .replace(
            'id="annotateModeButton" class="header-button active"',
            f'id="annotateModeButton" class="header-button" onclick="location.href=\'/annotate/{quote(project, safe="")}\'"',
        )
        .replace(
            'id="datasetButton" class="header-button"',
            f'id="datasetButton" class="header-button" onclick="location.href=\'/dataset/{quote(project, safe="")}\'"',
        )
        .replace(
            'id="modelButton" class="header-button"',
            f'id="modelButton" class="header-button" onclick="location.href=\'/model/{quote(project, safe="")}\'"',
        )
        .replace(
            'datasetButton.addEventListener("click", showEnterpriseNotice);',
            f'datasetButton.addEventListener("click", () => {{ location.href = "/dataset/{quote(project, safe="")}"; }});',
        )
        .replace(
            'modelButton.addEventListener("click", showEnterpriseNotice);',
            f'modelButton.addEventListener("click", () => {{ location.href = "/model/{quote(project, safe="")}"; }});',
        )
    )
    active_button = {
        "annotate": "annotateModeButton",
        "dataset": "datasetButton",
        "model": "modelButton",
    }.get(active_mode)
    if active_button:
        html = html.replace(
            f'id="{active_button}" class="header-button"',
            f'id="{active_button}" class="header-button active"',
        )
    if username:
        html = html.replace(
            '<body class="team-mode username-required" data-team-mode="true">',
            '<body class="team-mode" data-team-mode="true">',
            1,
        )
        html = html.replace(
            'window.yoloutilsUsername = "";',
            f'window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};',
            1,
        )
        html = html.replace(
            "window.yoloutilsUsernameReady = new Promise(resolve => {\n      window.yoloutilsResolveUsername = resolve;\n    });",
            "window.yoloutilsUsernameReady = Promise.resolve(window.yoloutilsUsername);\n    window.yoloutilsResolveUsername = () => {};",
            1,
        )
    if not team_mode_enabled():
        html = html.replace(
            '<button id="shareButton" class="header-button" title="分享当前页面或当前位置"><span class="header-icon">⇪</span><span>分享</span></button>',
            "",
            1,
        )
    user_script = (
        "<script>"
        f"window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};"
        f"window.yoloutilsProject = {json.dumps(project, ensure_ascii=False)};"
        f"window.yoloutilsOnlineUsers = {json.dumps(online_users, ensure_ascii=False)};"
        """
        (() => {
          const project = window.yoloutilsProject || "";
          if (!project) return;
          const nativeFetch = window.fetch.bind(window);
          window.fetch = (input, init) => {
            const url = typeof input === "string" ? input : input?.url;
            if (typeof url !== "string" || !url.startsWith("/annotate/api/")) {
              return nativeFetch(input, init);
            }
            const nextUrl = new URL(url, window.location.origin);
            if (!nextUrl.searchParams.has("project")) {
              nextUrl.searchParams.set("project", project);
            }
            if (typeof input === "string") {
              return nativeFetch(`${nextUrl.pathname}${nextUrl.search}`, init);
            }
            return nativeFetch(new Request(`${nextUrl.pathname}${nextUrl.search}`, input), init);
          };
        })();
        """
        "window.yoloutilsUsernameReady = Promise.resolve(window.yoloutilsUsername);"
        "try { localStorage.setItem('yoloutils-workstation-username', window.yoloutilsUsername); } catch (_) {}"
        "document.addEventListener('DOMContentLoaded', () => document.body.classList.remove('username-required'));"
        "</script>"
    )
    return html.replace("</head>", f"{user_script}</head>", 1)


def html_escape(value: str):
    return html.escape(value or "", quote=True)


def create_workstation():
    workstation = SiteWorkstation()
    workstation.host = os.environ.get("YOLOUTILS_HOST", workstation.host)
    try:
        workstation.port = int(os.environ.get("YOLOUTILS_PORT", workstation.port))
    except (TypeError, ValueError):
        pass
    dataset = os.environ.get("YOLOUTILS_DATASET")
    run = os.environ.get("YOLOUTILS_RUN")

    workstation.workspace = site_workspace()
    workstation.model_root = workstation.workspace
    workstation.log_root = workstation.workspace
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
    workstation.root_label = "根目录"
    workstation.class_groups = workstation._load_class_groups()
    workstation.classes_file = (
        workstation.class_groups[0]["path"] if workstation.class_groups else None
    )
    workstation.classes = (
        workstation.class_groups[0]["classes"] if workstation.class_groups else []
    )
    return workstation


def apply_project_workspace(workstation: Workstation, project: str):
    images_dir = project_images_workspace(project)
    workstation.workspace = images_dir if images_dir is not None else site_workspace()
    project_root = project_root_workspace(project) if images_dir is not None else None
    workstation.model_root = project_root if project_root is not None else workstation.workspace
    workstation.log_root = project_root if project_root is not None else workstation.workspace
    if project_root is not None and images_dir is not None:
        new_log = project_root / ".project.log"
        old_logs = [
            images_dir / ".yoloutils-workstation.log",
            project_root / ".yoloutils-workstation.log",
            project_root / ".yoloutils-upload.log",
        ]
        for old_log in old_logs:
            if not old_log.is_file() or old_log == new_log:
                continue
            try:
                with open(new_log, "a", encoding="utf-8") as target:
                    if new_log.stat().st_size:
                        target.write("\n")
                    target.write(f"[migrated] {old_log.name}\n")
                    target.write(old_log.read_text(encoding="utf-8", errors="replace"))
                old_log.unlink()
            except OSError:
                pass
    workstation.root_label = project_display_name(project) if images_dir is not None else "根目录"
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
    app.state.project_workspace_lock = asyncio.Lock()
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == "/"
            and "GET" in getattr(route, "methods", set())
        )
    ]

    @app.middleware("http")
    async def project_workspace_middleware(request: Request, call_next):
        project = request.query_params.get("project") or request.path_params.get("project") or request.cookies.get("current_project", "")
        async with app.state.project_workspace_lock:
            if project:
                apply_project_workspace(workstation, project)
            else:
                apply_project_workspace(workstation, "")
            response = await call_next(request)
        if project:
            response.set_cookie("current_project", project, httponly=True, samesite="lax")
        return response

    @app.get("/")
    def index(request: Request):
        try:
            username = current_username(request)
            if team_mode_enabled() and not username:
                return RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
            project = request.query_params.get("project") or request.cookies.get("current_project", "")
            if team_mode_enabled():
                write_user_project(username, project)
            apply_project_workspace(workstation, project)
            response = HTMLResponse(workstation_html(workstation, "annotate", project, username))
            if project:
                response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.get("/{project}")
    def index_project(request: Request, project: str):
        try:
            username = current_username(request)
            if team_mode_enabled() and not username:
                return RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
            if team_mode_enabled():
                write_user_project(username, project)
            apply_project_workspace(workstation, project)
            response = HTMLResponse(workstation_html(workstation, "annotate", project, username))
            response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app

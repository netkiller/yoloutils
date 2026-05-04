import os
import json
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

from workstation import Workstation


templates = Jinja2Templates(directory=TEMPLATES_DIR)


class SiteWorkstation(Workstation):
    def _directory_tree(self, path: Path):
        tree = super()._directory_tree(path)
        if path == self.workspace and path.name == "images":
            tree["name"] = "根目录"
        return tree


def site_workspace():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else PROJECT_ROOT


def read_online_users():
    path = site_workspace() / ".users"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    users = data.get("users", [])
    return [str(user).strip() for user in users if str(user).strip()] if isinstance(users, list) else []


def current_username(request: Request):
    username = (request.cookies.get("workstation_username") or "").strip()
    return username if username in read_online_users() else ""


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


def workstation_html(workstation: Workstation, active_mode: str = "annotate", project: str = "", username: str = ""):
    html = workstation._html()
    project_url = f"/project/{quote(project, safe='')}" if project else "/project"
    project_query = f"?project={quote(project, safe='')}" if project else ""
    project_button = (
        '<button id="projectButton" class="header-button" '
        f'title="项目" onclick="location.href=\'{project_url}\'">'
        '<span class="header-icon">▤</span><span>项目</span></button>'
    )
    html = (
        html.replace('"/api/', '"/annotate/api/')
        .replace("'/api/", "'/annotate/api/")
        .replace("`/api/", "`/annotate/api/")
        .replace('"/media', '"/annotate/media')
        .replace("'/media", "'/annotate/media")
        .replace("`/media", "`/annotate/media")
        .replace(
            "header { height: 48px;",
            "header { height: 56px;",
            1,
        )
        .replace(
            "padding: 0 16px; border-bottom:",
            "padding: 0 24px; border-bottom:",
            1,
        )
        .replace(
            '<a class="brand-link" href="https://www.netkiller.cn" target="_blank" rel="noopener noreferrer">Yolo Workstation</a>',
            '<a class="brand-link" href="https://www.netkiller.cn" target="_blank" rel="noopener noreferrer">Yolo Workstation</a>'
            f'<span class="enterprise-link">{username}</span>'
            '<form method="post" action="/project/logout" style="margin:0"><button class="enterprise-link" type="submit">注销</button></form>',
            1,
        )
        .replace(
            "main { height: calc(100vh - 88px);",
            "main { height: calc(100vh - 96px);",
            1,
        )
        .replace(
            "body.console-open main { height: calc(100vh - 88px - var(--console-height, 160px) - 4px);",
            "body.console-open main { height: calc(100vh - 96px - var(--console-height, 160px) - 4px);",
            1,
        )
        .replace(
            '<button id="annotateModeButton"',
            f'{project_button}<button id="annotateModeButton"',
            1,
        )
        .replace(
            'id="annotateModeButton" class="header-button active"',
            f'id="annotateModeButton" class="header-button" onclick="location.href=\'/annotate/{project_query}\'"',
        )
        .replace(
            'id="datasetButton" class="header-button"',
            f'id="datasetButton" class="header-button" onclick="location.href=\'/dataset{project_query}\'"',
        )
        .replace(
            'id="trainButton" class="header-button"',
            f'id="trainButton" class="header-button" onclick="location.href=\'/train{project_query}\'"',
        )
        .replace(
            'datasetButton.addEventListener("click", showEnterpriseNotice);',
            f'datasetButton.addEventListener("click", () => {{ location.href = "/dataset{project_query}"; }});',
        )
        .replace(
            'trainButton.addEventListener("click", showEnterpriseNotice);',
            f'trainButton.addEventListener("click", () => {{ location.href = "/train{project_query}"; }});',
        )
        .replace(
            f'<button id="trainButton" class="header-button" onclick="location.href=\'/train{project_query}\'" title="训练"><span class="header-icon">▶</span><span>训练</span></button>',
            f'<button id="trainButton" class="header-button" onclick="location.href=\'/train{project_query}\'" title="训练"><span class="header-icon">▶</span><span>训练</span></button>'
            f'<button id="validateButton" class="header-button" title="验证" onclick="location.href=\'/validate{project_query}\'"><span class="header-icon">✓</span><span>验证</span></button>',
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
    user_script = (
        "<script>"
        f"window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};"
        "try { localStorage.setItem('yoloutils-workstation-username', window.yoloutilsUsername); } catch (_) {}"
        "document.addEventListener('DOMContentLoaded', () => document.body.classList.remove('username-required'));"
        "</script>"
    )
    return html.replace("</head>", f"{user_script}</head>", 1)


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
            username = current_username(request)
            if not username:
                return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
            project = request.query_params.get("project", "")
            apply_project_workspace(workstation, project)
            return HTMLResponse(workstation_html(workstation, "annotate", project, username))
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from routes.project import header_context, project_dir, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


@router.get("/help")
def help_page(request: Request):
    workspace = workspace_path()
    requested_project = request.query_params.get("project") or request.cookies.get("current_project", "")
    current_project = requested_project if project_dir(workspace, requested_project) else ""
    response = templates.TemplateResponse(
        request=request,
        name="help/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "help",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/help/install.html")
def help_install_page(request: Request):
    workspace = workspace_path()
    requested_project = request.query_params.get("project") or request.cookies.get("current_project", "")
    current_project = requested_project if project_dir(workspace, requested_project) else ""
    response = templates.TemplateResponse(
        request=request,
        name="help/install.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "help",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/help/manual.html")
def help_manual_page(request: Request):
    workspace = workspace_path()
    requested_project = request.query_params.get("project") or request.cookies.get("current_project", "")
    current_project = requested_project if project_dir(workspace, requested_project) else ""
    response = templates.TemplateResponse(
        request=request,
        name="help/manual.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "help",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response

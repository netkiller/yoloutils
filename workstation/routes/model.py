from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context
from routes.train import load_tasks as load_train_tasks, model_items as run_model_items
from routes.val import dataset_items, project_path, read_project_name, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def model_context(request: Request, current_project: str):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "model",
        "model_active": "overview",
        "current_project": current_project,
        "project_name": read_project_name(path) if path else "",
        "models": run_model_items(load_train_tasks(), current_project) if path else [],
        "datasets": dataset_items(path) if path else [],
        **header_context(request, workspace),
    }


@router.get("/model")
def model_index(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/model/{project}", status_code=status.HTTP_303_SEE_OTHER)
    current_project = request.cookies.get("current_project", "")
    response = templates.TemplateResponse(
        request=request,
        name="model/index.html",
        context=model_context(request, current_project),
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}")
def model_project(request: Request, project: str):
    response = templates.TemplateResponse(
        request=request,
        name="model/index.html",
        context=model_context(request, project),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response

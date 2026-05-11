from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context
from routes.validate import dataset_items, model_items, project_path, read_project_name, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


@router.get("/predict")
def predict(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/predict/{project}", status_code=status.HTTP_303_SEE_OTHER)
    workspace = workspace_path()
    current_project = request.cookies.get("current_project", "")
    path = project_path(workspace, current_project)
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "predict",
            "current_project": current_project,
            "project_name": read_project_name(path) if path else "",
            "models": model_items(path) if path else [],
            "datasets": dataset_items(path) if path else [],
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/predict/{project}")
def predict_with_project(request: Request, project: str):
    workspace = workspace_path()
    current_project = project
    path = project_path(workspace, current_project)
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "predict",
            "current_project": current_project,
            "project_name": read_project_name(path) if path else "",
            "models": model_items(path) if path else [],
            "datasets": dataset_items(path) if path else [],
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response

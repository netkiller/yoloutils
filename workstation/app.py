from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import RedirectResponse

from routes.annotate import create_annotate_app
from routes.dataset import router as dataset_router
from routes.project import current_username, router as project_router, team_mode_enabled, workspace_path
from routes.train import router as train_router
from routes.validate import router as validate_router


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(dataset_router)
app.include_router(project_router)
app.include_router(train_router)
app.include_router(validate_router)


@app.get("/", include_in_schema=False)
def index(request: Request):
    if team_mode_enabled():
        if not current_username(request, workspace_path()):
            return RedirectResponse(url="/login")
        return RedirectResponse(url="/team")
    return RedirectResponse(url="/project")


@app.get("/annotate", include_in_schema=False)
def annotate(request: Request):
    query = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(url=f"/annotate/{query}")


app.mount("/annotate", create_annotate_app())

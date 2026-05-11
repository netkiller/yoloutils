import base64
import binascii
import os
import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from routes.annotate import create_annotate_app
from routes.dataset import router as dataset_router
from routes.help import router as help_router
from routes.project import current_username, router as project_router, team_mode_enabled, workspace_path
from routes.predict import router as predict_router
from routes.resources import router as resources_router
from routes.train import router as train_router
from routes.validate import router as validate_router


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(dataset_router)
app.include_router(help_router)
app.include_router(project_router)
app.include_router(predict_router)
app.include_router(resources_router)
app.include_router(train_router)
app.include_router(validate_router)


def basic_auth_credentials():
    auth = os.environ.get("YOLOUTILS_AUTH", "").strip()
    if not auth:
        return None
    username, separator, password = auth.partition(":")
    if not separator or not username or not password:
        return None
    return username, password


def unauthorized_response():
    return Response(
        "Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Yolo Workstation", charset="UTF-8"'},
    )


@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    credentials = basic_auth_credentials()
    if not credentials:
        return await call_next(request)

    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return unauthorized_response()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return unauthorized_response()
    username, separator, password = decoded.partition(":")
    if not separator:
        return unauthorized_response()
    expected_username, expected_password = credentials
    if not (
        secrets.compare_digest(username, expected_username)
        and secrets.compare_digest(password, expected_password)
    ):
        return unauthorized_response()
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index(request: Request):
    if team_mode_enabled():
        if not current_username(request, workspace_path()):
            return RedirectResponse(url="/login")
        return RedirectResponse(url="/team")
    return RedirectResponse(url="/project")


@app.get("/annotate", include_in_schema=False)
def annotate(request: Request):
    project = request.query_params.get("project")
    if project:
        return RedirectResponse(url=f"/annotate/{project}")
    return RedirectResponse(url="/annotate/")


app.mount("/annotate", create_annotate_app())

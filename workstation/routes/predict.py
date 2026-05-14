import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context
from routes.val import project_path, read_project_name, run_model_items, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def predict_root(path: Path):
    root = path / "predict"
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    return root


def safe_upload_name(filename: str):
    raw = Path((filename or "").replace("\\", "/")).name
    stem = Path(raw).stem.strip() or "media"
    suffix = Path(raw).suffix.lower()
    if suffix not in MEDIA_EXTS:
        return ""
    allowed = []
    for char in stem:
        allowed.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(allowed)[:80] + suffix


def resolve_model(path: Path, model: str):
    model_path = (path / model).resolve()
    if not model_path.is_file() or not is_inside(model_path, path):
        return None
    return model_path


async def save_upload_files(path: Path, files: list[UploadFile], capture_kind: str = ""):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    upload_dir = predict_root(path) / "uploads" / run_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, file in enumerate(files, start=1):
        name = safe_upload_name(file.filename or f"{capture_kind or 'media'}_{index}.jpg")
        if not name:
            continue
        target = upload_dir / name
        content = await file.read()
        if not content:
            continue
        target.write_bytes(content)
        saved.append(target)
    return run_name, upload_dir, saved


def output_items(project: str, project_dir: Path, run_dir: Path):
    items = []
    if not run_dir.is_dir():
        return items
    for file in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not file.is_file() or file.suffix.lower() not in MEDIA_EXTS:
            continue
        relative = file.relative_to(project_dir).as_posix()
        items.append(
            {
                "name": file.name,
                "type": "video" if file.suffix.lower() in VIDEO_EXTS else "image",
                "url": f"/model/{project}/predict/files/{relative}",
            }
        )
    return items


def run_predict(project: str, path: Path, model_path: Path, upload_dir: Path, run_name: str):
    run_dir = path / "predict-runs" / run_name
    if shutil.which("yolo") is None:
        return {
            "ok": False,
            "error": "yolo 命令不存在，请先安装 ultralytics。",
            "command": "",
            "outputs": [],
        }
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model_path}",
        f"source={upload_dir}",
        f"project={path / 'predict-runs'}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
    ]
    result = subprocess.run(
        command,
        cwd=workspace_path(),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "error": "" if result.returncode == 0 else (result.stderr or result.stdout or "预测失败"),
        "command": " ".join(str(part) for part in command),
        "outputs": output_items(project, path, run_dir),
    }


def predict_context(request: Request, current_project: str, result: dict | None = None):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "model",
        "model_active": "predict",
        "current_project": current_project,
        "project_name": read_project_name(path) if path else "",
        "models": run_model_items(path) if path else [],
        "result": result,
        **header_context(request, workspace),
    }


@router.get("/model/predict")
@router.get("/predict")
def predict(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    current_project = request.cookies.get("current_project", "")
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, current_project),
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/predict")
@router.get("/model/predict/{project}")
@router.get("/predict/{project}")
def predict_with_project(request: Request, project: str):
    if request.url.path.startswith("/model/predict/"):
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, project),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/model/{project}/predict")
@router.post("/model/predict/{project}")
@router.post("/predict/{project}")
async def run_predict_with_project(request: Request, project: str):
    workspace = workspace_path()
    path = project_path(workspace, project)
    result = {"ok": False, "error": "项目不存在", "outputs": []}
    if path:
        form = await request.form()
        model_path = resolve_model(path, str(form.get("model") or ""))
        files = [
            value
            for key, value in form.multi_items()
            if key == "media" and hasattr(value, "filename") and hasattr(value, "read")
        ]
        if model_path is None:
            result = {"ok": False, "error": "请选择有效模型", "outputs": []}
        elif not files:
            result = {"ok": False, "error": "请上传图片或视频", "outputs": []}
        else:
            run_name, upload_dir, saved = await save_upload_files(path, files, str(form.get("capture_kind") or ""))
            if not saved:
                result = {"ok": False, "error": "没有可用的图片或视频文件", "outputs": []}
            else:
                result = run_predict(project, path, model_path, upload_dir, run_name)
                result["saved"] = [item.name for item in saved]
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, project, result),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/predict/files/{file_path:path}")
@router.get("/model/predict/{project}/files/{file_path:path}")
@router.get("/predict/{project}/files/{file_path:path}")
def predict_file(project: str, file_path: str):
    workspace = workspace_path()
    path = project_path(workspace, project)
    if path is None:
        return RedirectResponse(url="/model/predict", status_code=status.HTTP_303_SEE_OTHER)
    file = (path / file_path).resolve()
    allowed_roots = ((path / "predict-runs").resolve(), (path / "predict" / "uploads").resolve())
    if not file.is_file() or not any(file == root or root in file.parents for root in allowed_roots):
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    return FileResponse(file)

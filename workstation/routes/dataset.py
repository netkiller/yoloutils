import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from routes.project import header_context


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def count_split(split_dir: Path):
    if not split_dir.is_dir():
        return {"images": 0, "labels": 0}
    images = sum(1 for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    labels = sum(1 for path in split_dir.rglob("*.txt") if path.is_file())
    return {"images": images, "labels": labels}


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


def project_name(project_dir: Path):
    meta = project_dir / ".project"
    if not meta.is_file():
        return project_dir.name
    try:
        import json

        data = json.loads(meta.read_text(encoding="utf-8"))
        return str(data.get("name") or project_dir.name)
    except (OSError, ValueError):
        return project_dir.name


def current_project_from_request(request: Request, fallback: str = ""):
    return fallback or request.cookies.get("current_project", "")


def image_files(root: Path):
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def copy_image_with_label(source: Path, source_root: Path, target_root: Path):
    relative = source.relative_to(source_root)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    label = source.with_suffix(".txt")
    if label.is_file():
        shutil.copy2(label, target.with_suffix(".txt"))


def build_dataset(workspace: Path, project: str, name: str, val_percent: int, test_percent: int):
    name = (name or "").strip()
    if not name or not DATASET_NAME_PATTERN.match(name):
        return None, "数据集名称只能包含字母、数字、点、下划线和连字符"
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
        return None, "val 和 test 百分比之和不能超过 100"

    project_path = project_dir(workspace, project)
    if project_path is None or not project_path.is_dir():
        return None, "项目不存在"

    images_root = project_path / "images"
    dataset_dir = project_path / "datasets" / name
    if dataset_dir.exists():
        return None, "数据集已存在"

    files = image_files(images_root)
    total = len(files)
    test_count = round(total * test_percent / 100)
    val_count = round(total * val_percent / 100)
    test_files = files[:test_count]
    val_files = files[test_count : test_count + val_count]
    train_files = files[test_count + val_count :]

    for split, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source in split_files:
            copy_image_with_label(source, images_root, split_dir)

    return {
        "path": str(dataset_dir),
        "total": total,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }, None


def dataset_dir(workspace: Path, project: str, name: str):
    project_path = project_dir(workspace, project)
    if project_path is None or not project_path.is_dir():
        return None
    datasets_root = (project_path / "datasets").resolve()
    path = (datasets_root / name).resolve()
    if path == datasets_root or not is_inside(path, datasets_root) or not path.is_dir():
        return None
    return path


def zip_dataset(path: Path):
    temp = tempfile.NamedTemporaryFile(prefix=f"{path.name}-", suffix=".zip", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix().lower()):
            if file.is_file():
                archive.write(file, file.relative_to(path).as_posix())
    return temp_path


def split_image_items(path: Path):
    items = {}
    for split in ("train", "val", "test"):
        split_dir = path / split
        files = image_files(split_dir)
        items[split] = [
            {
                "name": file.relative_to(split_dir).as_posix(),
                "media": f"/dataset/{path.parent.parent.name}/{path.name}/media/{split}/{file.relative_to(split_dir).as_posix()}",
                "label": file.with_suffix(".txt").is_file(),
            }
            for file in files
        ]
    return items


def dataset_items(workspace: Path, project: str = ""):
    datasets = []
    if not workspace.is_dir():
        return datasets

    for project_dir in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
        if project and project_dir.name != project:
            continue
        datasets_dir = project_dir / "datasets"
        if not project_dir.is_dir() or not datasets_dir.is_dir():
            continue
        for dataset_dir in sorted(datasets_dir.iterdir(), key=lambda item: item.name.lower()):
            if not dataset_dir.is_dir():
                continue
            splits = {
                "train": count_split(dataset_dir / "train"),
                "val": count_split(dataset_dir / "val"),
                "test": count_split(dataset_dir / "test"),
            }
            total_images = sum(split["images"] for split in splits.values())
            total_labels = sum(split["labels"] for split in splits.values())
            if total_images:
                train_percent = splits["train"]["images"] / total_images * 100
                val_percent = splits["val"]["images"] / total_images * 100
                test_percent = splits["test"]["images"] / total_images * 100
            else:
                train_percent = val_percent = test_percent = 0
            train_end = train_percent
            val_end = train_percent + val_percent
            chart_style = (
                f"conic-gradient(#1667c7 0 {train_end:.2f}%, "
                f"#16a34a {train_end:.2f}% {val_end:.2f}%, "
                f"#f59e0b {val_end:.2f}% 100%)"
                if total_images
                else "conic-gradient(#e2e8f0 0 100%)"
            )
            datasets.append(
                {
                    "name": dataset_dir.name,
                    "path": dataset_dir,
                    "project": project_name(project_dir),
                    "project_dir": project_dir.name,
                    "splits": splits,
                    "total_images": total_images,
                    "total_labels": total_labels,
                    "chart_style": chart_style,
                    "chart_segments": [
                        {"name": "train", "count": splits["train"]["images"], "percent": round(train_percent)},
                        {"name": "val", "count": splits["val"]["images"], "percent": round(val_percent)},
                        {"name": "test", "count": splits["test"]["images"], "percent": round(test_percent)},
                    ],
                }
            )
    return datasets


def dataset_summary(path: Path, project: str, name: str):
    splits = {
        "train": count_split(path / "train"),
        "val": count_split(path / "val"),
        "test": count_split(path / "test"),
    }
    return {
        "name": name,
        "project_dir": project,
        "project": project_name(path.parent.parent),
        "path": path,
        "splits": splits,
        "total_images": sum(split["images"] for split in splits.values()),
        "total_labels": sum(split["labels"] for split in splits.values()),
        "files": split_image_items(path),
    }


@router.get("/dataset")
def dataset(request: Request, project: str = ""):
    workspace = workspace_path()
    current_project = current_project_from_request(request, project)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace, current_project),
            "active_page": "dataset",
            "current_project": current_project,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/{name}")
def dataset_detail(request: Request, project: str, name: str):
    workspace = workspace_path()
    path = dataset_dir(workspace, project, name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/detail.html",
        context={
            "request": request,
            "workspace": workspace,
            "dataset": dataset_summary(path, project, name),
            "active_page": "dataset",
            "current_project": project,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/{name}/media/{split}/{file_path:path}")
def dataset_media(project: str, name: str, split: str, file_path: str):
    path = dataset_dir(workspace_path(), project, name)
    if path is None or split not in {"train", "val", "test"}:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    root = (path / split).resolve()
    image = (root / file_path).resolve()
    if not is_inside(image, root) or not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return FileResponse(image)


@router.post("/dataset")
async def create_dataset(request: Request):
    workspace = workspace_path()
    try:
        payload = await request.json()
        current_project = current_project_from_request(request, str(payload.get("project", "")))
        if not current_project:
            return JSONResponse({"ok": False, "error": "请先进入项目"}, status_code=400)
        result, error = build_dataset(
            workspace,
            current_project,
            str(payload.get("name", "")),
            int(payload.get("val_percent", 0) or 0),
            int(payload.get("test_percent", 0) or 0),
        )
        if error:
            return JSONResponse({"ok": False, "error": error}, status_code=400)
        return {"ok": True, **result}
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "创建数据集失败"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/dataset/{project}/{name}/download")
def download_dataset(project: str, name: str):
    path = dataset_dir(workspace_path(), project, name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    zip_path = zip_dataset(path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{name}.zip",
        background=BackgroundTask(lambda file: Path(file).unlink(missing_ok=True), zip_path),
    )

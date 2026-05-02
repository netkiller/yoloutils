import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def count_split(split_dir: Path):
    if not split_dir.is_dir():
        return {"images": 0, "labels": 0}
    images = sum(1 for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    labels = sum(1 for path in split_dir.rglob("*.txt") if path.is_file())
    return {"images": images, "labels": labels}


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


def dataset_items(workspace: Path):
    datasets = []
    if not workspace.is_dir():
        return datasets

    for project_dir in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
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
            datasets.append(
                {
                    "name": dataset_dir.name,
                    "path": dataset_dir,
                    "project": project_name(project_dir),
                    "project_dir": project_dir.name,
                    "splits": splits,
                    "total_images": total_images,
                    "total_labels": total_labels,
                }
            )
    return datasets


@router.get("/dataset")
def dataset(request: Request):
    workspace = workspace_path()
    return templates.TemplateResponse(
        request=request,
        name="dataset.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace),
            "active_page": "dataset",
        },
    )

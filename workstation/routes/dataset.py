import os
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from routes.project import header_context
from routes.resources import find_resource, read_resources


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ANNOTATE_DIR = "annotate"
DEPLOY_MODES = {"full": "全量", "incremental": "增量", "sync": "两端同步"}
DEPLOY_TARGETS = {"local": "本地", "remote": "远程"}
deploy_lock = threading.Lock()


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


def project_classes_file(project_path: Path):
    candidates = [project_path / "classes.txt", project_path / ANNOTATE_DIR / "classes.txt"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def build_dataset(workspace: Path, project: str, name: str, val_percent: int, test_percent: int):
    name = (name or "").strip()
    if not name or not DATASET_NAME_PATTERN.match(name):
        return None, "数据集名称只能包含字母、数字、点、下划线和连字符"
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
        return None, "val 和 test 百分比之和不能超过 100"

    project_path = project_dir(workspace, project)
    if project_path is None or not project_path.is_dir():
        return None, "项目不存在"

    images_root = project_path / ANNOTATE_DIR
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

    classes_file = project_classes_file(project_path)
    if classes_file:
        shutil.copy2(classes_file, dataset_dir / "classes.txt")

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


def deploy_root(project_path: Path):
    path = project_path / ".dataset-deploy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def deploy_tasks_file(project_path: Path):
    return deploy_root(project_path) / "tasks.json"


def read_deploy_tasks(project_path: Path):
    path = deploy_tasks_file(project_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    return [task for task in tasks if isinstance(task, dict)]


def write_deploy_tasks(project_path: Path, tasks: list[dict]):
    deploy_tasks_file(project_path).write_text(
        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_deploy_task(project_path: Path, task_id: str, **updates):
    with deploy_lock:
        tasks = read_deploy_tasks(project_path)
        for task in tasks:
            if task.get("id") == task_id:
                task.update(updates)
                task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        write_deploy_tasks(project_path, tasks)


def append_deploy_log(log_path: Path, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8", errors="replace") as output:
        output.write(f"[{timestamp}] {message}\n")


def dataset_select_items(workspace: Path, project: str):
    return [{"name": item["name"], "path": item["path"]} for item in dataset_items(workspace, project)]


def deploy_target_label(task: dict):
    if task.get("target_type") == "remote":
        return task.get("resource_name") or "远程服务器"
    return "本地"


def deploy_task_view(task: dict):
    return {
        **task,
        "mode_label": DEPLOY_MODES.get(task.get("mode"), task.get("mode", "")),
        "target_label": deploy_target_label(task),
        "target_type_label": DEPLOY_TARGETS.get(task.get("target_type"), task.get("target_type", "")),
    }


def remote_target(resource: dict, target_path: str):
    return f"{resource.get('username')}@{resource.get('host')}:{target_path}"


def remote_shell_path(target_path: str):
    if target_path == "~":
        return "$HOME"
    if target_path.startswith("~/"):
        return "$HOME/" + shlex.quote(target_path[2:])
    return shlex.quote(target_path)


def rsync_ssh_args(resource: dict, key_file: Path | None = None):
    args = ["ssh", "-p", str(resource.get("port") or 22), "-o", "StrictHostKeyChecking=no"]
    if key_file:
        args.extend(["-i", str(key_file)])
    return " ".join(args)


def prepare_remote_auth(resource: dict, temp_files: list[Path]):
    key_file = None
    if resource.get("use_private_key") and resource.get("private_key"):
        temp = tempfile.NamedTemporaryFile(prefix="dataset-rsync-key-", delete=False)
        key_file = Path(temp.name)
        temp.write(resource["private_key"].encode("utf-8"))
        temp.close()
        key_file.chmod(0o600)
        temp_files.append(key_file)
        return [], key_file, ""
    password = resource.get("password") or ""
    if password:
        sshpass = shutil.which("sshpass")
        if not sshpass:
            return [], None, "远程服务器使用密码认证，但本机未安装 sshpass，无法执行 rsync。请改用私钥或安装 sshpass。"
        return [sshpass, "-p", password], None, ""
    return [], None, ""


def run_command(command: list[str], log_path: Path):
    display = list(command)
    if display and Path(display[0]).name == "sshpass" and len(display) > 2:
        display[2] = "******"
    append_deploy_log(log_path, "$ " + " ".join(display))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        with log_path.open("a", encoding="utf-8", errors="replace") as output:
            output.write(line)
    return process.wait()


def remote_path_exists(resource: dict, target_path: str):
    try:
        import paramiko
        from routes.resources import ssh_connect_kwargs
    except ImportError:
        return False, "当前 Python 环境未安装 paramiko，无法检查远程目录。"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        command = f"test -e {remote_shell_path(target_path)}"
        _, stdout, _ = client.exec_command(command, timeout=10)
        return stdout.channel.recv_exit_status() == 0, ""
    except Exception as error:
        return False, f"检查远程目录失败：{error}"
    finally:
        client.close()


def remote_prepare(resource: dict, target_path: str, overwrite: bool):
    try:
        import paramiko
        from routes.resources import ssh_connect_kwargs
    except ImportError:
        return "当前 Python 环境未安装 paramiko，无法准备远程目录。"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        if overwrite:
            command = f"rm -rf {remote_shell_path(target_path)} && mkdir -p {remote_shell_path(target_path)}"
        else:
            command = f"mkdir -p {remote_shell_path(target_path)}"
        _, stdout, stderr = client.exec_command(command, timeout=60)
        code = stdout.channel.recv_exit_status()
        if code != 0:
            return stderr.read().decode("utf-8", errors="replace") or "远程目录准备失败"
        return ""
    except Exception as error:
        return f"准备远程目录失败：{error}"
    finally:
        client.close()


def build_rsync_commands(task: dict, source: Path, resource: dict | None, log_path: Path):
    source_arg = str(source) + "/"
    target_path = str(task["target_path"]).strip()
    mode = task["mode"]
    temp_files: list[Path] = []
    prefix: list[str] = []
    ssh_key = None
    auth_error = ""
    if task["target_type"] == "remote":
        assert resource is not None
        prefix, ssh_key, auth_error = prepare_remote_auth(resource, temp_files)
        if auth_error:
            return [], temp_files, auth_error
        target_arg = remote_target(resource, target_path.rstrip("/") + "/")
        base = [*prefix, "rsync", "-az", "-e", rsync_ssh_args(resource, ssh_key)]
        reverse_source = remote_target(resource, target_path.rstrip("/") + "/")
    else:
        target_arg = str(Path(target_path).expanduser()) + "/"
        base = ["rsync", "-az"]
        reverse_source = target_arg

    if mode == "full":
        return [[*base, "--delete", source_arg, target_arg]], temp_files, ""
    if mode == "incremental":
        return [[*base, "--ignore-existing", source_arg, target_arg]], temp_files, ""
    if mode == "sync":
        return [
            [*base, source_arg, target_arg],
            [*base, reverse_source, source_arg],
        ], temp_files, ""
    append_deploy_log(log_path, f"未知部署方式: {mode}")
    return [], temp_files, "未知部署方式"


def run_deploy_task(project_path: Path, task: dict):
    log_path = Path(task["log_path"])
    source = Path(task["source_path"])
    task_id = task["id"]
    temp_files: list[Path] = []
    try:
        update_deploy_task(project_path, task_id, status="进行中", progress=5)
        append_deploy_log(log_path, f"开始部署数据集 {task['dataset']}")
        if not source.is_dir():
            raise RuntimeError("源数据集目录不存在")

        resource = None
        if task["target_type"] == "remote":
            resource = find_resource(workspace_path(), task.get("resource_id", ""))
            if resource is None:
                raise RuntimeError("远程服务器不存在")
            exists, error = remote_path_exists(resource, task["target_path"])
            if error:
                raise RuntimeError(error)
            if task["mode"] == "full" and exists and not task.get("overwrite"):
                raise RuntimeError("目标目录已存在，请勾选覆盖删除后重新部署")
            prepare_error = remote_prepare(resource, task["target_path"], task["mode"] == "full" and bool(task.get("overwrite")))
            if prepare_error:
                raise RuntimeError(prepare_error)
        else:
            target = Path(task["target_path"]).expanduser()
            if task["mode"] == "full" and target.exists():
                if not task.get("overwrite"):
                    raise RuntimeError("目标目录已存在，请勾选覆盖删除后重新部署")
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

        update_deploy_task(project_path, task_id, progress=20)
        commands, temp_files, error = build_rsync_commands(task, source, resource, log_path)
        if error:
            raise RuntimeError(error)
        for index, command in enumerate(commands, start=1):
            append_deploy_log(log_path, f"执行 rsync ({index}/{len(commands)})")
            code = run_command(command, log_path)
            if code != 0:
                raise RuntimeError(f"rsync 退出码 {code}")
            update_deploy_task(project_path, task_id, progress=20 + round(index / len(commands) * 70))
        update_deploy_task(project_path, task_id, status="完成", progress=100)
        append_deploy_log(log_path, "部署完成")
    except Exception as error:
        update_deploy_task(project_path, task_id, status="失败", progress=100, error=str(error))
        append_deploy_log(log_path, f"部署失败：{error}")
    finally:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def create_deploy_task(project_path: Path, dataset_path: Path, dataset_name: str, form):
    target_type = (form.get("target_type", "local") or "local").strip()
    mode = (form.get("mode", "incremental") or "incremental").strip()
    if target_type not in DEPLOY_TARGETS:
        return None, "目标类型不正确"
    if mode not in DEPLOY_MODES:
        return None, "部署方式不正确"
    resource_id = (form.get("resource_id", "") or "").strip()
    resource = find_resource(workspace_path(), resource_id) if target_type == "remote" else None
    if target_type == "remote" and resource is None:
        return None, "请选择远程服务器"
    target_path = (form.get("target_path", "") or "").strip() or f"~/datasets/{dataset_name}"
    if "\n" in target_path or "\r" in target_path:
        return None, "部署位置不能包含换行"
    task_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    log_path = deploy_root(project_path) / f"{task_id}.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task = {
        "id": task_id,
        "dataset": dataset_name,
        "source_path": str(dataset_path),
        "target_type": target_type,
        "target_path": target_path,
        "resource_id": resource_id,
        "resource_name": resource.get("name", "") if resource else "",
        "mode": mode,
        "overwrite": (form.get("overwrite", "") or "").lower() in {"1", "true", "yes", "on"},
        "status": "排队中",
        "progress": 0,
        "error": "",
        "log_path": str(log_path),
        "created_at": now,
        "updated_at": now,
    }
    with deploy_lock:
        tasks = read_deploy_tasks(project_path)
        tasks.insert(0, task)
        write_deploy_tasks(project_path, tasks)
    append_deploy_log(log_path, "部署任务已创建")
    threading.Thread(target=run_deploy_task, args=(project_path, task), daemon=True, name=f"dataset-deploy-{task_id}").start()
    return task, ""


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


def read_classes_for_dataset(dataset_path: Path):
    path = dataset_path / "classes.txt"
    if not path.is_file():
        path = project_classes_file(dataset_path.parent.parent) or dataset_path.parent.parent / "classes.txt"
    if not path.is_file():
        return {"exists": False, "class_names": [], "text": ""}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "exists": True,
        "class_names": [line.strip() for line in text.splitlines() if line.strip()],
        "text": text,
    }


def class_annotations(path: Path, class_names: list[str]):
    counts = {}
    for label_file in sorted(path.rglob("*.txt"), key=lambda item: item.as_posix().lower()):
        try:
            lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                index = int(float(parts[0]))
            except ValueError:
                continue
            counts[index] = counts.get(index, 0) + 1

    max_index = max(counts.keys(), default=-1)
    total_classes = max(len(class_names), max_index + 1)
    rows = []
    max_count = max(counts.values(), default=0)
    total_annotations = sum(counts.values())
    for index in range(total_classes):
        count = counts.get(index, 0)
        rows.append(
            {
                "index": index,
                "name": class_names[index] if index < len(class_names) else f"class_{index}",
                "count": count,
                "percent": round(count / max_count * 100, 2) if max_count else 0,
            }
        )
    def scale_label(value: float):
        return str(int(value)) if value.is_integer() else f"{value:.1f}"

    scale = [scale_label(max_count * ratio) for ratio in (1, 0.75, 0.5, 0.25, 0)] if max_count else ["0"]
    return {
        "rows": rows,
        "total_classes": total_classes,
        "total_annotations": total_annotations,
        "total_annotations_label": f"{total_annotations:,}",
        "scale": scale,
    }


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
                    "updated_at": dataset_dir.stat().st_mtime,
                    "updated_date": datetime.fromtimestamp(dataset_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
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
    classes = read_classes_for_dataset(path)
    annotations = class_annotations(path, classes["class_names"])
    return {
        "name": name,
        "project_dir": project,
        "project": project_name(path.parent.parent),
        "path": path,
        "splits": splits,
        "total_images": sum(split["images"] for split in splits.values()),
        "total_labels": sum(split["labels"] for split in splits.values()),
        "files": split_image_items(path),
        "classes": classes,
        "annotations": annotations,
    }


@router.get("/dataset")
def dataset(request: Request, project: str = ""):
    workspace = workspace_path()
    current_project = current_project_from_request(request, project)
    if current_project:
        return RedirectResponse(url=f"/dataset/{current_project}", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace, ""),
            "active_page": "dataset",
            "current_project": "",
            "current_project_name": "",
            **header_context(request, workspace),
        },
    )
    return response


@router.get("/dataset/{project}")
def dataset_with_project(request: Request, project: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace, project),
            "active_page": "dataset",
            "current_project": project,
            "current_project_name": project_name(current_project_path) if current_project_path and current_project_path.is_dir() else project,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/deploy")
@router.get("/dataset/{project}/deploy/{name}")
def dataset_deploy(request: Request, project: str, name: str = ""):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    datasets = dataset_select_items(workspace, project)
    selected = next((item for item in datasets if item["name"] == name), datasets[0] if datasets else None)
    tasks = [deploy_task_view(task) for task in read_deploy_tasks(current_project_path)]
    response = templates.TemplateResponse(
        request=request,
        name="dataset/deploy.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": datasets,
            "selected_dataset": selected,
            "tasks": tasks,
            "resources": read_resources(workspace),
            "active_page": "dataset",
            "current_project": project,
            "current_project_name": project_name(current_project_path),
            "deploy_modes": DEPLOY_MODES,
            "deploy_targets": DEPLOY_TARGETS,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/dataset/{project}/deploy")
@router.post("/dataset/{project}/deploy/{name}")
async def create_dataset_deploy(request: Request, project: str, name: str = ""):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    form = await request.form()
    dataset_name = name or str(form.get("dataset", "") or "").strip()
    path = dataset_dir(workspace, project, dataset_name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    task, error = create_deploy_task(current_project_path, path, dataset_name, form)
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=400)
    return RedirectResponse(
        url=f"/dataset/{project}/deploy/{dataset_name}#task-{task['id']}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/dataset/{project}/deploy/tasks/{task_id}")
def dataset_deploy_task(request: Request, project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    task = next((item for item in read_deploy_tasks(current_project_path) if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    log_path = Path(task.get("log_path", ""))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    response = templates.TemplateResponse(
        request=request,
        name="dataset/deploy_task.html",
        context={
            "request": request,
            "workspace": workspace,
            "task": deploy_task_view(task),
            "log_text": log_text,
            "active_page": "dataset",
            "current_project": project,
            "current_project_name": project_name(current_project_path),
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/deploy/tasks/{task_id}/log")
def dataset_deploy_task_log(project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    task = next((item for item in read_deploy_tasks(current_project_path) if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    log_path = Path(task.get("log_path", ""))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return {
        "ok": True,
        "status": task.get("status", ""),
        "progress": task.get("progress", 0),
        "error": task.get("error", ""),
        "log": log_text,
    }


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

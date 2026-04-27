import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse
    import uvicorn
    from PIL import ExifTags, Image
except ImportError:
    FastAPI = None
    HTTPException = None
    Query = None
    Request = None
    FileResponse = None
    HTMLResponse = None
    uvicorn = None
    ExifTags = None
    Image = None

try:
    from . import Common
except ImportError:
    from __init__ import Common


class Workstation:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        daemon: bool = False,
    ):
        self.host = host
        self.port = port
        self.daemon = daemon
        self.workspace = None
        self.dataset = None
        self.run = None
        self.requested_classes_file = None
        self.open_browser = False
        self.team_mode = False
        self.share_url = ""
        self.presence = {}
        self.locks = {}
        self.classes_file = None
        self.class_groups = []
        self.classes = []

    def main(
        self,
        workspace: str,
        dataset: str = None,
        run: str = None,
        classes_file: str = None,
        open_browser: bool = False,
        team_mode: bool = False,
        share_url: str = "",
    ):
        if FastAPI is None or uvicorn is None:
            print("缺少依赖: fastapi/uvicorn，请先安装: pip install fastapi uvicorn")
            return

        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            print(f"workspace 目录不存在: {self.workspace}")
            return
        self.dataset = Path(dataset).expanduser().resolve() if dataset else None
        self.run = Path(run).expanduser().resolve() if run else None
        self.requested_classes_file = classes_file
        self.open_browser = open_browser
        self.team_mode = team_mode
        self.share_url = share_url.strip().rstrip("/") if share_url else ""

        if self.daemon:
            self._start_daemon()
            return

        try:
            self.class_groups = self._load_class_groups()
        except FileNotFoundError as error:
            print(error)
            return
        self.classes_file = self.class_groups[0]["path"] if self.class_groups else None
        self.classes = self.class_groups[0]["classes"] if self.class_groups else []
        app = self._create_app()

        url = self._server_url()
        print(f"Yolo Workstation: {url}")
        print(f"Workspace: {self.workspace}")
        if self.dataset:
            print(f"Dataset: {self.dataset}")
        if self.run:
            print(f"Run: {self.run}")
        if self.classes_file:
            print(f"Classes: {self.classes_file}")
        else:
            print("Classes: 未找到 classes.txt")
        if self.open_browser:
            self._start_browser_opener(url)
            print("Browser: opening")
        uvicorn.run(app, host=self.host, port=self.port)

    def _server_url(self):
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def _share_url(self):
        if self.share_url:
            return self.share_url
        host = self._public_ip() or self.host
        if host in ("", "0.0.0.0", "::", "127.0.0.1", "localhost"):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def _public_ip(self):
        for url in (
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://icanhazip.com",
        ):
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    ip = response.read().decode("utf-8", errors="ignore").strip()
                if ip and len(ip) <= 64 and " " not in ip:
                    return ip
            except OSError:
                continue
        return ""

    def _lan_ip(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"

    def _start_browser_opener(self, url: str):
        def open_and_keep_alive():
            time.sleep(1)
            self._open_app_window(url)
            while True:
                try:
                    request = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Yolo-Workstation-Headless/1.0"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        response.read(2048)
                except Exception:
                    pass
                time.sleep(60)

        threading.Thread(target=open_and_keep_alive, daemon=True, name="yoloutils-browser-opener").start()

    def _open_app_window(self, url: str):
        if sys.platform == "darwin":
            for app_name in ("Google Chrome", "Microsoft Edge", "Brave Browser", "Chromium"):
                try:
                    result = subprocess.run(
                        ["open", "-na", app_name, "--args", f"--app={url}", "--new-window"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=3,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if result.returncode == 0:
                    return
        commands = [
            ["google-chrome", f"--app={url}", "--new-window"],
            ["chrome", f"--app={url}", "--new-window"],
            ["chromium", f"--app={url}", "--new-window"],
            ["chromium-browser", f"--app={url}", "--new-window"],
            ["microsoft-edge", f"--app={url}", "--new-window"],
            ["brave-browser", f"--app={url}", "--new-window"],
        ]
        for command in commands:
            try:
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue

    def _pid_file(self):
        return self.workspace / ".yoloutils-workstation.pid"

    def _log_file(self):
        return self.workspace / ".yoloutils-workstation.log"

    def _is_process_running(self, pid: int):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _existing_pid(self):
        pid_file = self._pid_file()
        if not pid_file.exists():
            return None
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
        if self._is_process_running(pid):
            return pid
        pid_file.unlink(missing_ok=True)
        return None

    def _daemon_command(self):
        args = list(sys.argv)
        args = [arg for arg in args if arg not in ("-d", "--daemon")]

        executable = Path(args[0])
        if executable.name == "__main__.py" and executable.parent.name == "yoloutils":
            return [sys.executable, "-m", "netkiller.yoloutils"] + args[1:]
        if executable.exists():
            return [sys.executable] + args
        return args

    def _start_daemon(self):
        pid = self._existing_pid()
        if pid is not None:
            print(f"Yolo Workstation 已在后台运行: pid={pid}")
            print(f"Yolo Workstation: http://{self.host}:{self.port}")
            print(f"PID: {self._pid_file()}")
            print(f"LOG: {self._log_file()}")
            return

        log_file = self._log_file()
        with open(log_file, "ab") as output:
            process = subprocess.Popen(
                self._daemon_command(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                cwd=os.getcwd(),
                start_new_session=True,
                close_fds=True,
            )
        self._pid_file().write_text(str(process.pid), encoding="utf-8")
        print(f"Yolo Workstation 已后台启动: pid={process.pid}")
        print(f"Yolo Workstation: http://{self.host}:{self.port}")
        print(f"PID: {self._pid_file()}")
        print(f"LOG: {log_file}")

    def _safe_path(self, relative_path: str = ""):
        relative_path = relative_path or ""
        path = (self.workspace / relative_path).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        return path

    def _relative(self, path: Path):
        return path.relative_to(self.workspace).as_posix()

    def _is_inside_workspace(self, path: Path):
        try:
            path.relative_to(self.workspace)
            return True
        except ValueError:
            return False

    def _resolve_classes_file(self, classes_file: str):
        path = Path(classes_file).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"classes.txt 文件不存在: {path}")
        return path

    def _find_classes_files(self):
        if self.requested_classes_file:
            return [self._resolve_classes_file(self.requested_classes_file)]

        files = sorted(self.workspace.rglob("classes.txt"), key=lambda item: self._relative(item).lower())
        root_classes = self.workspace / "classes.txt"
        if root_classes in files:
            files.remove(root_classes)
            files.insert(0, root_classes)
        return files

    def _load_classes_file(self, classes_file: Path):
        with open(classes_file, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def _load_class_groups(self):
        groups = []
        for classes_file in self._find_classes_files():
            groups.append(
                {
                    "path": classes_file,
                    "classes_file": self._relative(classes_file)
                    if self._is_inside_workspace(classes_file)
                    else str(classes_file),
                    "classes": self._load_classes_file(classes_file),
                }
            )
        return groups

    def _max_class_count(self):
        return max([len(group["classes"]) for group in self.class_groups] + [len(self.classes), 0])

    def _is_image(self, path: Path):
        return path.is_file() and path.name.lower().endswith(Common.image_exts)

    def _image_files(self):
        return sorted(
            (path for path in self.workspace.rglob("*") if self._is_image(path)),
            key=lambda path: self._relative(path).lower(),
        )

    def _directory_tree(self, path: Path):
        children = [
            self._directory_tree(child)
            for child in sorted(path.iterdir(), key=lambda item: item.name.lower())
            if child.is_dir()
        ]
        images = [item for item in path.iterdir() if self._is_image(item)]
        direct_complete = all(
            not self._is_damaged_image(image)
            and self._validate_label_file(image.with_suffix(".txt")) == "valid"
            for image in images
        )
        has_images = bool(images) or any(child["has_images"] for child in children)
        complete = has_images and direct_complete and all(
            child["complete"] for child in children if child["has_images"]
        )
        return {
            "name": path.name if path != self.workspace else self.workspace.name,
            "path": "" if path == self.workspace else self._relative(path),
            "has_images": has_images,
            "complete": complete,
            "children": children,
        }

    def _list_files(self, directory: str):
        path = self._safe_path(directory)
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="directory not found")

        files = []
        for item in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            if not self._is_image(item):
                continue
            label_file = item.with_suffix(".txt")
            label_status = self._validate_label_file(label_file)
            damaged = self._is_damaged_image(item)
            label_count = self._label_count(label_file) if label_status == "valid" else 0
            files.append(
                {
                    "name": item.name,
                    "path": self._relative(item),
                    "label": self._relative(label_file) if label_file.exists() else None,
                    "label_status": label_status,
                    "label_count": label_count,
                    "damaged": damaged,
                }
            )
        return files

    def _label_count(self, label_file: Path):
        try:
            return len([line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()])
        except (OSError, UnicodeDecodeError):
            return 0

    def _validate_label_file(self, label_file: Path):
        if not label_file.exists():
            return "missing"

        try:
            lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines()]
        except UnicodeDecodeError:
            return "invalid"

        lines = [line for line in lines if line]
        if not lines:
            return "empty"

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                return "invalid"
            try:
                class_id = int(parts[0])
                [float(value) for value in parts[1:]]
            except ValueError:
                return "invalid"
            if class_id < 0 or class_id >= self._max_class_count():
                return "invalid"
        return "valid"

    def _is_damaged_image(self, image_file: Path):
        if Image is None:
            return False
        try:
            with Image.open(image_file) as image:
                image.verify()
            return False
        except Exception:
            return True

    def _statistics(self):
        images = self._image_files()
        result = {
            "workspace": str(self.workspace),
            "images": len(images),
            "images_damaged": 0,
            "txt_total": 0,
            "txt_missing": 0,
            "txt_empty": 0,
            "txt_invalid": 0,
            "txt_valid": 0,
        }

        for image in images:
            if self._is_damaged_image(image):
                result["images_damaged"] += 1
            status = self._validate_label_file(image.with_suffix(".txt"))
            if status == "missing":
                result["txt_missing"] += 1
            elif status == "empty":
                result["txt_total"] += 1
                result["txt_empty"] += 1
            elif status == "invalid":
                result["txt_total"] += 1
                result["txt_invalid"] += 1
            else:
                result["txt_total"] += 1
                result["txt_valid"] += 1

        result["txt_problem"] = (
            result["txt_missing"] + result["txt_empty"] + result["txt_invalid"]
        )
        result["txt_invalid_total"] = result["txt_empty"] + result["txt_invalid"]
        result["classes"] = len(self.classes)
        result["classes_files"] = len(self.class_groups)
        return result

    def _read_annotation(self, image_path: str):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")

        label_file = path.with_suffix(".txt")
        boxes = []
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        cx, cy, width, height = [float(value) for value in parts[1:]]
                    except ValueError:
                        continue
                    boxes.append(
                        {
                            "line": line_number,
                            "class_id": class_id,
                            "label": self.classes[class_id]
                            if 0 <= class_id < len(self.classes)
                            else str(class_id),
                            "cx": cx,
                            "cy": cy,
                            "width": width,
                            "height": height,
                        }
                    )
        return {
            "image": self._relative(path),
            "label_file": self._relative(label_file) if label_file.exists() else None,
            "boxes": boxes,
        }

    def _write_annotation(self, image_path: str, boxes, client_id: str = "", username: str = ""):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        if not isinstance(boxes, list):
            raise HTTPException(status_code=400, detail="boxes must be a list")
        relative_path = self._relative(path)
        if self.team_mode:
            lock = self._lock_for(relative_path)
            if lock and lock["client_id"] != client_id:
                raise HTTPException(status_code=423, detail=f"文件已被 {lock['username']} 锁定")

        lines = []
        for box in boxes:
            try:
                class_id = int(box["class_id"])
                cx = float(box["cx"])
                cy = float(box["cy"])
                width = float(box["width"])
                height = float(box["height"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(status_code=400, detail="invalid box")
            if class_id < 0 or (self._max_class_count() and class_id >= self._max_class_count()):
                raise HTTPException(status_code=400, detail="invalid class_id")
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

        label_file = path.with_suffix(".txt")
        if lines:
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            label_file.write_text("", encoding="utf-8")
        if self.team_mode:
            self._append_operation_log(username or "未命名", relative_path, lines)
        if self.team_mode and client_id:
            self._release_lock(relative_path, client_id)
        return self._read_annotation(image_path)

    def _append_operation_log(self, username: str, image_path: str, lines):
        timestamp = datetime.now().isoformat(timespec="seconds")
        summary = "; ".join(lines) if lines else "清空标注"
        message = f"[{timestamp}] {username} 保存 {image_path}: {summary}\n"
        try:
            with open(self._log_file(), "a", encoding="utf-8") as file:
                file.write(message)
        except OSError:
            pass

    def _json_value(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return [self._json_value(item) for item in value]
        if isinstance(value, list):
            return [self._json_value(item) for item in value]
        return str(value)

    def _read_exif(self, image_path: str):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")

        stat = path.stat()
        info = {
            "文件": self._relative(path),
            "文件大小": stat.st_size,
            "修改时间": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        exif = {}

        with Image.open(path) as image:
            info["格式"] = image.format
            info["宽"] = image.width
            info["高"] = image.height
            info["模式"] = image.mode
            raw_exif = image.getexif()
            for tag_id, value in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif[tag] = self._json_value(value)

        return {"info": info, "exif": exif}

    def _online_count(self):
        now = time.time()
        self.presence = {
            client_id: info
            for client_id, info in self.presence.items()
            if now - info["seen_at"] <= 12
        }
        return len(self.presence)

    def _online_users(self):
        self._online_count()
        return [
            {"client_id": client_id, "username": info["username"]}
            for client_id, info in sorted(
                self.presence.items(),
                key=lambda item: item[1]["username"].lower(),
            )
        ]

    def _lock_for(self, image_path: str):
        lock = self.locks.get(image_path)
        if not lock:
            return None
        if time.time() - lock["seen_at"] > 180:
            self.locks.pop(image_path, None)
            return None
        return lock

    def _acquire_lock(self, image_path: str, client_id: str, username: str):
        if not self.team_mode:
            return {"team_mode": False, "locked": False, "owner": None}
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id required")
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        relative_path = self._relative(path)
        lock = self._lock_for(relative_path)
        if lock and lock["client_id"] != client_id:
            return {"team_mode": True, "locked": True, "owner": lock}
        self.locks[relative_path] = {
            "client_id": client_id,
            "username": username or "未命名",
            "path": relative_path,
            "seen_at": time.time(),
        }
        return {"team_mode": True, "locked": False, "owner": self.locks[relative_path]}

    def _release_lock(self, image_path: str, client_id: str):
        lock = self._lock_for(image_path)
        if lock and lock["client_id"] == client_id:
            self.locks.pop(image_path, None)
        return {"released": True}

    def _leave_presence(self, client_id: str):
        if client_id:
            self.presence.pop(client_id, None)
        return {"online": self._online_count(), "users": self._online_users()}

    def _read_logs(self, lines: int = 200):
        log_file = self._log_file()
        if not log_file.exists():
            return {
                "file": str(log_file),
                "lines": [
                    "当前会话没有后台日志文件。",
                    "使用 -d/--daemon 后，日志会写入 .yoloutils-workstation.log。",
                ],
            }
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            return {"file": str(log_file), "lines": [f"读取日志失败: {error}"]}
        return {"file": str(log_file), "lines": content[-lines:]}

    def _create_app(self):
        app = FastAPI(title="Yolo Workstation")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(self._html())

        @app.get("/api/config")
        def config():
            return {"team_mode": self.team_mode, "share_url": self._share_url()}

        @app.get("/api/tree")
        def tree():
            return self._directory_tree(self.workspace)

        @app.get("/api/files")
        def files(directory: str = Query(default="")):
            return {"directory": directory, "files": self._list_files(directory)}

        @app.get("/api/classes")
        def classes():
            return {
                "classes_file": self.class_groups[0]["classes_file"] if self.class_groups else None,
                "classes": self.classes,
                "class_groups": [
                    {
                        "classes_file": group["classes_file"],
                        "classes": group["classes"],
                    }
                    for group in self.class_groups
                ],
            }

        @app.get("/api/statistics")
        def statistics():
            return self._statistics()

        @app.post("/api/presence")
        async def presence(request: Request):
            payload = await request.json()
            client_id = str(payload.get("client_id", "")).strip()
            username = str(payload.get("username", "")).strip() or "独立用户"
            if not client_id:
                raise HTTPException(status_code=400, detail="client_id required")
            self.presence[client_id] = {
                "username": username,
                "seen_at": time.time(),
            }
            return {"online": self._online_count(), "users": self._online_users()}

        @app.post("/api/presence/leave")
        async def leave_presence(request: Request):
            payload = await request.json()
            return self._leave_presence(str(payload.get("client_id", "")).strip())

        @app.post("/api/lock")
        async def lock(request: Request):
            payload = await request.json()
            return self._acquire_lock(
                payload.get("path", ""),
                str(payload.get("client_id", "")).strip(),
                str(payload.get("username", "")).strip(),
            )

        @app.post("/api/lock/release")
        async def release_lock(request: Request):
            payload = await request.json()
            return self._release_lock(
                str(payload.get("path", "")).strip(),
                str(payload.get("client_id", "")).strip(),
            )

        @app.get("/api/logs")
        def logs(lines: int = Query(default=200, ge=1, le=1000)):
            return self._read_logs(lines)

        @app.get("/api/annotation")
        def annotation(path: str):
            return self._read_annotation(path)

        @app.post("/api/annotation")
        async def save_annotation(request: Request):
            payload = await request.json()
            return self._write_annotation(
                payload.get("path", ""),
                payload.get("boxes", []),
                str(payload.get("client_id", "")).strip(),
                str(payload.get("username", "")).strip(),
            )

        @app.get("/api/exif")
        def exif(path: str):
            return self._read_exif(path)

        @app.get("/media")
        def media(path: str):
            file_path = self._safe_path(path)
            if not self._is_image(file_path):
                raise HTTPException(status_code=404, detail="image not found")
            return FileResponse(file_path)

        return app

    def _html(self):
        return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yolo Workstation</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f5f7fa; }
    header { height: 48px; display: grid; grid-template-columns: minmax(160px, 1fr) auto minmax(160px, 1fr); align-items: center; gap: 16px; padding: 0 16px; border-bottom: 1px solid #d9e2ec; background: #fff; font-weight: 650; }
    .header-title { min-width: 0; display: flex; align-items: center; gap: 8px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .brand-link { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; color: #1f2933; text-decoration: none; }
    .brand-link:hover { color: #1d4ed8; }
    .enterprise-link { flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center; height: 24px; padding: 0 8px; border: 1px solid #d9e2ec; border-radius: 6px; color: #334e68; background: #fff; font-size: 12px; line-height: 1; text-decoration: none; }
    .enterprise-link:hover { background: #e6f0ff; color: #243b53; }
    .header-modes { display: flex; align-items: center; justify-content: center; gap: 8px; }
    .header-actions { min-width: 0; display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
    .header-button { width: auto; min-width: 62px; height: 30px; display: inline-flex; align-items: center; justify-content: center; gap: 5px; padding: 0 9px; border: 1px solid #d9e2ec; border-radius: 6px; background: #fff; color: #334e68; font-size: 12px; line-height: 1; white-space: nowrap; }
    .header-button:hover { background: #e6f0ff; color: #243b53; }
    .header-button.active { border-color: #2563eb; background: #dbeafe; color: #1d4ed8; }
    .header-icon { font-size: 15px; line-height: 1; }
    button:disabled { cursor: not-allowed; opacity: .42; }
    button:disabled:hover { background: transparent; color: inherit; }
    main { height: calc(100vh - 88px); display: grid; grid-template-columns: minmax(360px, 520px) 1px minmax(420px, 1fr) 220px; overflow: hidden; }
    body.console-open main { height: calc(100vh - 248px); }
    main.tree-hidden { grid-template-columns: 280px 1px minmax(420px, 1fr) 220px; }
    main.right-hidden { grid-template-columns: minmax(360px, 520px) 1px minmax(420px, 1fr) 0; }
    main.tree-hidden.right-hidden { grid-template-columns: 280px 1px minmax(420px, 1fr) 0; }
    main.viewer-focus { display: flex; }
    main.viewer-focus .left-panel, main.viewer-focus .main-splitter, main.viewer-focus .right-panel { display: none; }
    main.viewer-focus section.viewer { flex: 1 1 auto; width: 100%; border-right: 0; }
    aside, section { min-width: 0; overflow: auto; border-right: 1px solid #d9e2ec; background: #fff; }
    section.viewer { background: #f5f7fa; display: flex; flex-direction: column; }
    h2 { margin: 0; padding: 12px 14px; font-size: 13px; border-bottom: 1px solid #e4e7eb; background: #f8fafc; }
    .pane-header { flex: 0 0 34px; height: 34px; min-height: 34px; max-height: 34px; display: flex; align-items: center; justify-content: space-between; gap: 6px; padding: 0 8px 0 12px; overflow: hidden; border-bottom: 1px solid #e4e7eb; background: #f8fafc; }
    .pane-header h2 { flex: 1 1 auto; min-width: 0; padding: 0; overflow: hidden; border: 0; background: transparent; line-height: 1; white-space: nowrap; text-overflow: ellipsis; }
    .viewer-header { position: relative; }
    .viewer-header h2 { padding-right: 76px; }
    .pane-title-icon { display: inline-block; width: 16px; margin-right: 6px; color: #52606d; font-size: 15px; line-height: 1; font-weight: 500; text-align: center; }
    .icon-button { flex: 0 0 24px; width: 24px; min-width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; padding: 0; text-align: center; border-radius: 5px; font-size: 14px; color: #52606d; }
    .icon-button:hover { background: #e6f0ff; color: #243b53; }
    .viewer-tools { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
    .tool-button { width: auto; min-width: 34px; height: 28px; padding: 0 8px; display: inline-flex; align-items: center; justify-content: center; gap: 6px; border: 1px solid #d9e2ec; border-radius: 6px; background: #fff; color: #334e68; font-size: 12px; line-height: 1; white-space: nowrap; }
    .tool-button:hover { background: #e6f0ff; color: #243b53; }
    .tool-button.active { border-color: #2563eb; background: #dbeafe; color: #1d4ed8; }
    .shortcut-hint { margin-left: 2px; color: #7b8794; font-size: 11px; font-weight: 650; }
    .left-panel { display: grid; grid-template-columns: minmax(120px, 46%) 1px minmax(160px, 1fr); overflow: hidden; border-right: 1px solid #d9e2ec; background: #fff; }
    .left-panel.tree-hidden { display: block; }
    .left-panel.tree-hidden .files-pane { height: 100%; }
    .left-pane { min-width: 0; overflow: hidden; background: #fff; display: flex; flex-direction: column; }
    .left-panel.tree-hidden .tree-pane, .left-panel.tree-hidden .vertical-splitter { display: none; }
    .vertical-splitter { cursor: col-resize; background: #d9e2ec; }
    .vertical-splitter:hover, .vertical-splitter.dragging { background: #bcccdc; }
    .main-splitter { cursor: col-resize; background: #d9e2ec; }
    .main-splitter:hover, .main-splitter.dragging { background: #bcccdc; }
    .tree, .files, .labels, .exif, .collaboration { padding: 8px; }
    .tree { flex: 4 1 80%; min-height: 0; overflow: auto; }
    .files { flex: 1 1 auto; min-height: 0; overflow: auto; }
    .collaboration { display: none; flex: 1 1 20%; min-height: 72px; overflow: hidden; border-top: 1px solid #d9e2ec; background: #fff; }
    body.team-mode .collaboration { display: block; }
    .collaboration-header { height: 34px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 8px 0 12px; border-bottom: 1px solid #e4e7eb; background: #f8fafc; color: #1f2933; font-size: 13px; font-weight: 650; }
    .online-count { min-width: 20px; color: #1f2933; font-size: 13px; font-weight: 750; text-align: right; }
    .collaboration-users { max-height: calc(100% - 34px); overflow: auto; display: grid; gap: 2px; padding: 6px 8px; color: #52606d; font-size: 12px; line-height: 1.35; }
    .collaboration-user { display: flex; align-items: center; gap: 6px; min-height: 22px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .collaboration-user::before { content: ""; flex: 0 0 7px; width: 7px; height: 7px; border-radius: 50%; background: #22c55e; }
    .lock-note { color: #b45309; font-size: 12px; }
    button { width: 100%; border: 0; background: transparent; text-align: left; padding: 7px 8px; border-radius: 6px; cursor: pointer; color: #243b53; font: inherit; }
    button:hover, button.active { background: #e6f0ff; }
    .tree-node { position: relative; }
    .tree-row { display: flex; align-items: center; min-width: 0; }
    .tree-children { margin-left: 14px; padding-left: 8px; border-left: 1px solid #d9e2ec; }
    .tree-toggle { flex: 0 0 22px; width: 22px; min-width: 22px; height: 28px; display: inline-flex; align-items: center; justify-content: center; padding: 0; color: #7b8794; font-size: 13px; }
    .tree-toggle-placeholder { flex: 0 0 22px; width: 22px; min-width: 22px; }
    .tree-select { flex: 1 1 auto; display: flex; align-items: center; gap: 7px; min-width: 0; padding: 6px 8px; }
    .tree-icon { flex: 0 0 auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 10px; line-height: 1; font-weight: 700; }
    .tree-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .file-meta { display: block; margin-top: 2px; color: #7b8794; font-size: 12px; }
    .file-valid { color: #166534; }
    .file-valid .file-meta { color: #15803d; }
    .file-invalid { color: #991b1b; }
    .file-invalid .file-meta { color: #dc2626; }
    .file-valid.active { background: #dcfce7; }
    .file-invalid.active { background: #fee2e2; }
    .files button { font-size: 12px; line-height: 1.25; }
    .file-name { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .file-icon { flex: 0 0 auto; color: #52606d; font-size: 14px; line-height: 1; }
    .file-name-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pane-tools { flex: 0 0 auto; display: flex; align-items: center; gap: 4px; }
    .canvas-wrap { flex: 1; min-height: 0; display: flex; align-items: flex-start; justify-content: flex-start; padding: 16px; overflow: auto; }
    .viewer-zoom { position: absolute; left: 50%; top: 50%; min-width: 64px; transform: translate(-50%, -50%); text-align: center; color: #52606d; font-size: 12px; font-weight: 700; line-height: 1; pointer-events: none; }
    .zoom-reset-hint { margin-left: 8px; color: #7b8794; font-size: 11px; font-weight: 650; }
    #canvas { margin: auto; max-width: none; max-height: none; background: #fff; box-shadow: 0 1px 8px rgba(31, 41, 51, .18); }
    #canvas.annotating { cursor: crosshair; }
    #canvas.panning { cursor: grabbing; }
    .empty { color: #7b8794; padding: 16px; }
    .label-source { margin: 8px 0 4px; color: #52606d; font-size: 12px; font-weight: 700; overflow-wrap: anywhere; }
    .label-row { width: 100%; display: flex; align-items: center; gap: 8px; padding: 6px 4px; font-size: 13px; }
    .label-row.active { background: #dbeafe; color: #1d4ed8; }
    .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    .right-panel { display: flex; flex-direction: column; overflow: hidden; }
    main.right-hidden .right-panel { display: none; }
    .right-pane { min-height: 80px; overflow: auto; }
    .top-info-pane { flex: 0 0 40%; min-height: 180px; display: flex; flex-direction: column; overflow: hidden; }
    .labels-pane { flex: 1 1 55%; min-height: 42px; overflow: hidden; display: flex; flex-direction: column; }
    .labels { flex: 1 1 auto; min-height: 0; overflow: auto; }
    .histogram-pane { flex: 1 1 45%; min-height: 42px; overflow: hidden; display: flex; flex-direction: column; }
    .right-panel.histogram-collapsed .labels-pane { flex-grow: 0; }
    .right-panel.histogram-collapsed .histogram-pane { flex: 0 0 34px; }
    .right-panel.histogram-collapsed .histogram-splitter { display: none; }
    .right-panel.histogram-collapsed .histogram { display: none; }
    .right-pane.exif-pane { flex: 1 1 auto; }
    .right-panel.exif-collapsed .top-info-pane { flex: 1 1 auto; }
    .right-panel.exif-collapsed .splitter { display: none; }
    .right-panel.exif-collapsed .exif-pane { flex: 0 0 34px; min-height: 34px; overflow: hidden; }
    .right-panel.exif-collapsed .exif { display: none; }
    .histogram { flex: 1 1 auto; min-height: 0; padding: 8px; }
    #histogramCanvas { width: 100%; height: 100%; display: block; background: #fff; border: 1px solid #e4e7eb; border-radius: 4px; }
    .splitter { flex: 0 0 1px; cursor: row-resize; background: #d9e2ec; }
    .splitter:hover, .splitter.dragging { background: #bcccdc; }
    .histogram-splitter { flex: 0 0 1px; cursor: row-resize; background: #d9e2ec; }
    .histogram-splitter:hover, .histogram-splitter.dragging { background: #bcccdc; }
    .kv-row { display: grid; grid-template-columns: minmax(76px, 38%) minmax(0, 1fr); gap: 8px; padding: 6px 4px; border-bottom: 1px solid #edf2f7; font-size: 12px; }
    .kv-key { color: #52606d; overflow-wrap: anywhere; }
    .kv-value { color: #1f2933; overflow-wrap: anywhere; }
    footer { height: 40px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 16px; border-top: 1px solid #d9e2ec; background: #fff; font-size: 13px; color: #52606d; }
    .stats-left { min-width: 0; display: flex; align-items: center; overflow: hidden; white-space: nowrap; }
    .breadcrumb { min-width: 0; display: flex; align-items: center; overflow: hidden; color: #52606d; }
    .breadcrumb button { width: auto; min-width: 0; flex: 0 1 auto; padding: 3px 4px; border-radius: 4px; color: #1f2933; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .breadcrumb button:hover { background: #e6f0ff; }
    .breadcrumb-home { flex: 0 0 auto; padding-right: 6px; color: #52606d; font-size: 17px; line-height: 1; }
    .breadcrumb-separator { flex: 0 0 auto; color: #9aa5b1; padding: 0 2px; }
    .breadcrumb-file { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #1f2933; padding-left: 4px; }
    .stats-right { flex: 0 0 auto; display: flex; align-items: center; justify-content: flex-end; gap: 18px; white-space: nowrap; }
    .stat-icon { display: inline-block; margin-right: 4px; color: #52606d; font-size: 14px; line-height: 1; }
    .stat strong { color: #1f2933; font-weight: 700; }
    .stat.warn strong { color: #b45309; }
    .stat.bad strong { color: #be123c; }
    .footer-button { width: auto; height: 26px; display: inline-flex; align-items: center; justify-content: center; gap: 5px; padding: 0 8px; border: 1px solid #d9e2ec; border-radius: 6px; background: #fff; color: #334e68; font-size: 12px; line-height: 1; white-space: nowrap; }
    .footer-button:hover, .footer-button.active { background: #e6f0ff; color: #243b53; }
    .console-panel { height: 160px; display: none; border-top: 1px solid #d9e2ec; background: #111827; color: #d1d5db; overflow: auto; }
    body.console-open .console-panel { display: block; }
    .console-log { margin: 0; padding: 10px 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.45; white-space: pre-wrap; }
    .shortcut-popover { position: fixed; top: 56px; right: 16px; z-index: 20; width: 300px; padding: 12px; border: 1px solid #d9e2ec; border-radius: 8px; background: #fff; box-shadow: 0 12px 28px rgba(31, 41, 51, .18); color: #243b53; }
    .shortcut-popover[hidden] { display: none; }
    .shortcut-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; font-size: 13px; font-weight: 700; }
    .shortcut-title button { width: 24px; height: 24px; padding: 0; display: inline-flex; align-items: center; justify-content: center; color: #52606d; }
    .shortcut-list { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 7px 12px; font-size: 12px; line-height: 1.35; }
    .shortcut-key { color: #1d4ed8; font-weight: 750; white-space: nowrap; }
    .shortcut-desc { color: #52606d; }
    .username-gate { position: fixed; inset: 0; z-index: 40; display: none; align-items: center; justify-content: center; background: rgba(15, 23, 42, .42); }
    body.username-required .username-gate { display: flex; }
    .username-dialog { width: 320px; padding: 18px; border-radius: 10px; background: #fff; box-shadow: 0 18px 45px rgba(15, 23, 42, .24); }
    .username-dialog h3 { margin: 0 0 12px; font-size: 16px; color: #1f2933; }
    .username-dialog input { width: 100%; height: 34px; padding: 0 10px; border: 1px solid #bcccdc; border-radius: 6px; font: inherit; }
    .username-dialog button { width: auto; height: 30px; margin-top: 12px; padding: 0 12px; border: 1px solid #2563eb; background: #2563eb; color: #fff; text-align: center; }
  </style>
</head>
<body class="__BODY_CLASS__" data-team-mode="__TEAM_MODE__">
  <header>
    <div class="header-title">
      <a class="brand-link" href="https://www.netkiller.cn" target="_blank" rel="noopener noreferrer">Yolo Workstation</a>
      <a class="enterprise-link" href="https://saas.netkiller.cn" target="_blank" rel="noopener noreferrer">企业版</a>
    </div>
    <div class="header-modes">
      <button id="annotateModeButton" class="header-button active" title="当前窗口：标注"><span class="header-icon">▧</span><span>标注</span></button>
      <button id="datasetButton" class="header-button" title="数据集"><span class="header-icon">▦</span><span>数据集</span></button>
      <button id="trainButton" class="header-button" title="训练"><span class="header-icon">▶</span><span>训练</span></button>
    </div>
    <div class="header-actions">
      <button id="autoAnnotate" class="header-button" title="自动标注激活"><span class="header-icon">◎</span><span>自动</span></button>
      <button id="editModeToggle" class="header-button" title="只读/打标切换"><span class="header-icon">◌</span><span>只读</span></button>
      <button id="shareButton" class="header-button" title="分享当前页面或当前位置"><span class="header-icon">⇪</span><span>分享</span></button>
      <button id="downloadImage" class="header-button" title="下载当前图片"><span class="header-icon">⇩</span><span>下载</span></button>
      <button id="queryButton" class="header-button" title="查询目录或当前文件列表"><span class="header-icon">⌕</span><span>查询</span></button>
      <button id="shortcutButton" class="header-button" title="查看快捷键"><span class="header-icon">⌘</span><span>快捷键</span></button>
    </div>
  </header>
  <div id="shortcutPopover" class="shortcut-popover" hidden>
    <div class="shortcut-title"><span>快捷键</span><button id="closeShortcuts" type="button" title="关闭">×</button></div>
    <div class="shortcut-list">
      <span class="shortcut-key">↑ / ↓</span><span class="shortcut-desc">在目录树中切换目录</span>
      <span class="shortcut-key">← / →</span><span class="shortcut-desc">切换上一张 / 下一张图片</span>
      <span class="shortcut-key">⌘M</span><span class="shortcut-desc">切换 box 半透明遮罩</span>
      <span class="shortcut-key">⌘S</span><span class="shortcut-desc">保存当前标注</span>
      <span class="shortcut-key">⌘D</span><span class="shortcut-desc">删除当前标注</span>
      <span class="shortcut-key">⌘R</span><span class="shortcut-desc">重置当前操作</span>
      <span class="shortcut-key">ESC</span><span class="shortcut-desc">还原图片缩放 / 关闭快捷键窗口</span>
      <span class="shortcut-key">滚轮</span><span class="shortcut-desc">快速放大或缩小图片</span>
      <span class="shortcut-key">双击图像标题</span><span class="shortcut-desc">隐藏或显示左右栏</span>
    </div>
  </div>
  <div id="usernameGate" class="username-gate">
    <form id="usernameForm" class="username-dialog">
      <h3>请输入用户名</h3>
      <input id="usernameInput" type="text" autocomplete="name" placeholder="用户名">
      <button id="usernameSubmit" type="submit">进入</button>
    </form>
  </div>
  <script>
    window.yoloutilsUsername = "";
    window.yoloutilsUsernameReady = new Promise(resolve => {
      window.yoloutilsResolveUsername = resolve;
    });
    window.yoloutilsSubmitUsername = event => {
      if (event) event.preventDefault();
      const input = document.getElementById("usernameInput");
      const name = (input?.value || "").trim();
      if (!name) {
        input?.focus();
        return;
      }
      window.yoloutilsUsername = name;
      try {
        localStorage.setItem("yoloutils-workstation-username", name);
      } catch (error) {
        // localStorage can be blocked in some app-window modes.
      }
      document.body.classList.remove("username-required");
      window.yoloutilsResolveUsername?.();
    };
    document.addEventListener("DOMContentLoaded", () => {
      const form = document.getElementById("usernameForm");
      const input = document.getElementById("usernameInput");
      try {
        input.value = localStorage.getItem("yoloutils-workstation-username") || "";
      } catch (error) {
        input.value = "";
      }
      form?.addEventListener("submit", window.yoloutilsSubmitUsername);
      if (document.body.classList.contains("username-required")) input?.focus();
    });
  </script>
  <main id="appMain">
    <div id="leftPanel" class="left-panel">
      <aside id="treePane" class="left-pane tree-pane">
        <div class="pane-header">
          <h2><span class="pane-title-icon">▰</span>目录</h2>
          <div class="pane-tools">
            <button id="reloadTree" class="icon-button" title="重载目录">↻</button>
            <button id="toggleTree" class="icon-button" title="隐藏目录栏">‹</button>
          </div>
        </div>
        <div id="tree" class="tree"></div>
        <div id="collaboration" class="collaboration">
          <div class="collaboration-header"><span><span class="pane-title-icon">◌</span>协作</span><strong id="onlineCount" class="online-count">-</strong></div>
          <div id="collaborationUsers" class="collaboration-users"></div>
        </div>
      </aside>
      <div id="leftSplitter" class="vertical-splitter" title="拖动调整目录和文件列表比例"></div>
      <aside class="left-pane files-pane">
        <div class="pane-header">
          <h2><span class="pane-title-icon">▯</span>文件</h2>
          <div class="pane-tools">
            <button id="toggleFileSort" class="icon-button" title="未达标排在上面">↑</button>
            <button id="showTree" class="icon-button" title="显示目录栏">☰</button>
          </div>
        </div>
        <div id="files" class="files"></div>
      </aside>
    </div>
    <div id="mainSplitter" class="main-splitter" title="拖动调整文件列表和图像区域比例"></div>
    <section class="viewer">
      <div id="viewerHeader" class="pane-header viewer-header">
        <h2 id="viewerTitle"><span class="pane-title-icon">▧</span>图像</h2>
        <div id="zoomIndicator" class="viewer-zoom">100%</div>
        <div class="viewer-tools">
          <button id="maskAnnotation" class="tool-button" title="切换 box 遮罩"><span class="header-icon">◧</span><span>遮罩</span><span class="shortcut-hint">⌘M</span></button>
          <button id="deleteAnnotation" class="tool-button" title="删除当前标注"><span class="header-icon">×</span><span>删除</span><span class="shortcut-hint">⌘D</span></button>
          <button id="resetAnnotation" class="tool-button" title="重新读取标注"><span class="header-icon">↺</span><span>重置</span><span class="shortcut-hint">⌘R</span></button>
          <button id="saveAnnotation" class="tool-button" title="保存当前标注"><span class="header-icon">✓</span><span>保存</span><span class="shortcut-hint">⌘S</span></button>
          <button id="showRight" class="icon-button" title="显示标签/信息栏">☰</button>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="canvas"></canvas><div id="empty" class="empty">请选择图片</div></div>
    </section>
    <aside id="rightPanel" class="right-panel">
      <div id="topInfoPane" class="right-pane top-info-pane">
        <div id="labelsPane" class="labels-pane">
          <div class="pane-header"><h2><span class="pane-title-icon">⌑</span>标签</h2><button id="hideRight" class="icon-button" title="隐藏标签/信息栏">›</button></div>
          <div id="labels" class="labels"></div>
        </div>
        <div id="histogramSplitter" class="histogram-splitter" title="拖动调整标签和直方图比例"></div>
        <div id="histogramPane" class="histogram-pane">
          <div class="pane-header"><h2><span class="pane-title-icon">▥</span>直方图</h2><button id="toggleHistogram" class="icon-button" title="折叠/展开直方图">⌄</button></div>
          <div id="histogram" class="histogram"><canvas id="histogramCanvas"></canvas></div>
        </div>
      </div>
      <div id="splitter" class="splitter" title="拖动调整标签和 EXIF 面板比例"></div>
      <div id="exifPane" class="right-pane exif-pane">
        <div class="pane-header"><h2><span class="pane-title-icon">※</span>信息</h2><button id="toggleExif" class="icon-button" title="折叠/展开信息">⌄</button></div>
        <div id="exif" class="exif"><div class="empty">请选择图片</div></div>
      </div>
    </aside>
  </main>
  <footer id="stats">
    <div class="stats-left">-</div>
    <div class="stats-right">
      <span class="stat"><span class="stat-icon">▧</span>图像 <strong>-</strong></span>
      <span class="stat"><span class="stat-icon">✓</span>已完成 <strong>-/-</strong></span>
      <span class="stat"><span class="stat-icon">▯</span>.txt <strong>-</strong></span>
      <span class="stat"><span class="stat-icon">⌑</span>classes.txt <strong>-</strong></span>
      <span class="stat bad"><span class="stat-icon">✕</span>损坏图像 <strong>-</strong></span>
      <span class="stat bad"><span class="stat-icon">!</span>无效 .txt <strong>-</strong></span>
      <button id="consoleToggle" class="footer-button" type="button"><span class="stat-icon">▤</span>控制台</button>
    </div>
  </footer>
  <section id="consolePanel" class="console-panel"><pre id="consoleLog" class="console-log">控制台未打开</pre></section>
  <script>
    const colors = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f"];
    const treeEl = document.getElementById("tree");
    const filesEl = document.getElementById("files");
    const appMain = document.getElementById("appMain");
    const leftPanel = document.getElementById("leftPanel");
    const leftSplitter = document.getElementById("leftSplitter");
    const mainSplitter = document.getElementById("mainSplitter");
    const toggleTree = document.getElementById("toggleTree");
    const reloadTree = document.getElementById("reloadTree");
    const showTree = document.getElementById("showTree");
    const toggleFileSort = document.getElementById("toggleFileSort");
    const labelsEl = document.getElementById("labels");
    const exifEl = document.getElementById("exif");
    const topInfoPane = document.getElementById("topInfoPane");
    const labelsPane = document.getElementById("labelsPane");
    const histogramPane = document.getElementById("histogramPane");
    const histogramSplitter = document.getElementById("histogramSplitter");
    const histogramCanvas = document.getElementById("histogramCanvas");
    const histogramCtx = histogramCanvas.getContext("2d");
    const splitter = document.getElementById("splitter");
    const rightPanel = document.getElementById("rightPanel");
    const hideRight = document.getElementById("hideRight");
    const showRight = document.getElementById("showRight");
    const autoAnnotate = document.getElementById("autoAnnotate");
    const editModeToggle = document.getElementById("editModeToggle");
    const maskAnnotation = document.getElementById("maskAnnotation");
    const deleteAnnotation = document.getElementById("deleteAnnotation");
    const resetAnnotation = document.getElementById("resetAnnotation");
    const saveAnnotation = document.getElementById("saveAnnotation");
    const toggleHistogram = document.getElementById("toggleHistogram");
    const toggleExif = document.getElementById("toggleExif");
    const shareButton = document.getElementById("shareButton");
    const downloadImage = document.getElementById("downloadImage");
    const queryButton = document.getElementById("queryButton");
    const shortcutButton = document.getElementById("shortcutButton");
    const shortcutPopover = document.getElementById("shortcutPopover");
    const closeShortcuts = document.getElementById("closeShortcuts");
    const annotateModeButton = document.getElementById("annotateModeButton");
    const datasetButton = document.getElementById("datasetButton");
    const trainButton = document.getElementById("trainButton");
    const statsEl = document.getElementById("stats");
    const onlineCount = document.getElementById("onlineCount");
    const collaborationUsers = document.getElementById("collaborationUsers");
    const consoleToggle = document.getElementById("consoleToggle");
    const consoleLog = document.getElementById("consoleLog");
    const usernameForm = document.getElementById("usernameForm");
    const usernameInput = document.getElementById("usernameInput");
    const usernameSubmit = document.getElementById("usernameSubmit");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const empty = document.getElementById("empty");
    const viewerTitle = document.getElementById("viewerTitle");
    const viewerHeader = document.getElementById("viewerHeader");
    const zoomIndicator = document.getElementById("zoomIndicator");
    let currentDir = "";
    let currentPath = "";
    let currentImage = null;
    let imageZoom = 1;
    let currentBoxes = [];
    let savedBoxes = [];
    let draftBox = null;
    let dragStart = null;
    let selectedBoxIndex = -1;
    let boxHitAreas = [];
    let canvasInteraction = null;
    let currentFiles = [];
    let fileSortDirection = "asc";
    let annotateMode = false;
    let maskEnabled = false;
    let boxesDirty = false;
    let classLabels = [];
    let selectedClassId = 0;
    let selectedClassLabel = "0";
    let teamMode = false;
    let username = "";
    let usernameReady = window.yoloutilsUsernameReady || Promise.resolve();
    let usernameGateBound = false;
    let lockedByOther = null;
    let shareUrl = window.location.origin;
    const collapsedDirs = new Set();
    let statisticsData = null;
    let histogramExpandedTopHeight = null;

    async function getJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[char]));
    }

    function hashColor(value) {
      let hash = 0;
      for (let index = 0; index < value.length; index += 1) {
        hash = ((hash << 5) - hash + value.charCodeAt(index)) | 0;
      }
      return colors[Math.abs(hash) % colors.length];
    }

    function renderTree(node, parent = treeEl) {
      const wrapper = document.createElement("div");
      wrapper.className = "tree-node";
      const row = document.createElement("div");
      row.className = "tree-row";
      const hasChildren = node.children.length > 0;
      const isCollapsed = collapsedDirs.has(node.path);
      if (hasChildren) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "tree-toggle";
        toggle.textContent = isCollapsed ? "▸" : "▾";
        toggle.title = isCollapsed ? "展开目录" : "折叠目录";
        toggle.onclick = event => {
          event.stopPropagation();
          if (collapsedDirs.has(node.path)) collapsedDirs.delete(node.path);
          else collapsedDirs.add(node.path);
          loadTree();
        };
        row.appendChild(toggle);
      } else {
        const placeholder = document.createElement("span");
        placeholder.className = "tree-toggle-placeholder";
        row.appendChild(placeholder);
      }
      const button = document.createElement("button");
      button.className = "tree-select";
      button.dataset.path = node.path;
      button.title = node.path || node.name || "/";
      const icon = document.createElement("span");
      icon.className = "tree-icon";
      icon.textContent = node.complete ? "[x]" : "[]";
      icon.style.color = hashColor(node.path || node.name || "/");
      const name = document.createElement("span");
      name.className = "tree-name";
      name.textContent = node.name || "/";
      button.appendChild(icon);
      button.appendChild(name);
      button.onclick = () => selectDir(node.path, button);
      row.appendChild(button);
      wrapper.appendChild(row);
      if (hasChildren) {
        const children = document.createElement("div");
        children.className = "tree-children";
        children.hidden = isCollapsed;
        wrapper.appendChild(children);
        for (const child of node.children) renderTree(child, children);
      }
      parent.appendChild(wrapper);
    }

    async function loadTree() {
      const tree = await getJson("/api/tree");
      treeEl.innerHTML = "";
      renderTree(tree);
      setActiveTreeButton(currentDir);
      return tree;
    }

    function findTreeButton(path) {
      return Array.from(document.querySelectorAll("#tree .tree-select"))
        .find(item => item.dataset.path === path);
    }

    function setActiveTreeButton(path, button = null) {
      document.querySelectorAll("#tree .tree-select").forEach(item => item.classList.remove("active"));
      const activeButton = button || findTreeButton(path);
      if (activeButton) activeButton.classList.add("active");
    }

    async function selectDir(path, button) {
      currentDir = path;
      setActiveTreeButton(path, button);
      const data = await getJson(`/api/files?directory=${encodeURIComponent(path)}`);
      currentFiles = data.files;
      renderStatistics();
      renderFiles();
      if (!currentFiles.some(file => file.path === currentPath)) {
        await selectFirstCurrentFile();
      }
    }

    function fileSortRank(file) {
      return !file.damaged && file.label_status === "valid" ? 1 : 0;
    }

    function sortedCurrentFiles() {
      const direction = fileSortDirection === "desc" ? -1 : 1;
      return [...currentFiles].sort((left, right) => {
        const rankDiff = (fileSortRank(left) - fileSortRank(right)) * direction;
        if (rankDiff !== 0) return rankDiff;
        return left.name.localeCompare(right.name, "zh-Hans-CN", {numeric: true, sensitivity: "base"});
      });
    }

    function fileIcon(name) {
      const extension = name.toLowerCase().slice(name.lastIndexOf("."));
      return {
        ".jpg": "▣",
        ".jpeg": "▧",
        ".png": "◩",
        ".bmp": "▥",
        ".webp": "◫",
        ".tif": "◬",
        ".tiff": "◭",
        ".heic": "◈",
        ".heif": "◆",
        ".avif": "◇",
      }[extension] || "▯";
    }

    function renderFiles() {
      filesEl.innerHTML = "";
      toggleFileSort.textContent = fileSortDirection === "asc" ? "↑" : "↓";
      toggleFileSort.title = fileSortDirection === "asc" ? "未达标排在上面" : "已达标排在上面";
      if (!currentFiles.length) {
        filesEl.innerHTML = '<div class="empty">没有图片</div>';
        return;
      }
      for (const file of sortedCurrentFiles()) {
        const row = document.createElement("button");
        const isInvalid = file.damaged || file.label_status === "empty" || file.label_status === "invalid";
        const isValid = !file.damaged && file.label_status === "valid";
        if (isInvalid) row.classList.add("file-invalid");
        if (isValid) row.classList.add("file-valid");
        const statusText = file.damaged
          ? "损坏图像"
          : file.label_status === "valid"
            ? `已标注 ${file.label_count || 0}`
            : file.label_status === "missing"
              ? "未标注"
              : file.label_status === "empty"
                ? "空 .txt"
                : "无效 .txt";
        row.innerHTML = `<span class="file-name"><span class="file-icon">${fileIcon(file.name)}</span><span class="file-name-text">${escapeHtml(file.name)}</span></span><span class="file-meta">${statusText}</span>`;
        row.dataset.path = file.path;
        row.onclick = () => selectImage(file.path, row);
        filesEl.appendChild(row);
      }
      if (currentPath) {
        const activeRow = Array.from(document.querySelectorAll("#files button"))
          .find(button => button.dataset.path === currentPath);
        if (activeRow) activeRow.classList.add("active");
      }
    }

    async function refreshCurrentFiles() {
      const data = await getJson(`/api/files?directory=${encodeURIComponent(currentDir)}`);
      currentFiles = data.files;
      renderStatistics();
      renderFiles();
    }

    function fileRow(path) {
      return Array.from(document.querySelectorAll("#files button"))
        .find(button => button.dataset.path === path);
    }

    async function selectAdjacentImage(direction) {
      const files = sortedCurrentFiles();
      if (!files.length) return;
      const currentIndex = files.findIndex(file => file.path === currentPath);
      const index = currentIndex < 0
        ? 0
        : (currentIndex + direction + files.length) % files.length;
      const path = files[index].path;
      await selectImage(path, fileRow(path));
    }

    function nextAnnotationFilePath(savedPath) {
      const files = sortedCurrentFiles().filter(file => file.path !== savedPath);
      if (!files.length) return "";
      const incomplete = files.find(file => fileSortRank(file) === 0);
      return (incomplete || files[0]).path;
    }

    async function selectFirstCurrentFile() {
      const [firstFile] = sortedCurrentFiles();
      if (!firstFile) {
        currentPath = "";
        currentImage = null;
        currentBoxes = [];
        savedBoxes = [];
        draftBox = null;
        dragStart = null;
        selectedBoxIndex = -1;
        canvasInteraction = null;
        setBoxesDirty(false);
        resetImageZoom();
        viewerTitle.innerHTML = '<span class="pane-title-icon">▧</span>图像';
        canvas.style.display = "none";
        empty.style.display = "block";
        exifEl.innerHTML = '<div class="empty">请选择图片</div>';
        return;
      }
      const row = Array.from(document.querySelectorAll("#files button"))
        .find(button => button.dataset.path === firstFile.path);
      await selectImage(firstFile.path, row);
    }

    async function selectImage(path, button) {
      document.querySelectorAll("#files button").forEach(item => item.classList.remove("active"));
      if (button) button.classList.add("active");
      viewerTitle.innerHTML = `<span class="pane-title-icon">▧</span>${escapeHtml(path)}`;
      currentPath = path;
      lockedByOther = null;
      renderStatistics();
      const annotation = await getJson(`/api/annotation?path=${encodeURIComponent(path)}`);
      currentBoxes = cloneBoxes(annotation.boxes);
      savedBoxes = cloneBoxes(annotation.boxes);
      draftBox = null;
      dragStart = null;
      selectedBoxIndex = -1;
      canvasInteraction = null;
      setBoxesDirty(false);
      if (teamMode) await acquireCurrentLock();
      loadExif(path);
      const image = new Image();
      image.onload = () => {
        currentImage = image;
        empty.style.display = "none";
        canvas.style.display = "block";
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        resetImageZoom();
        drawHistogram(image);
        redrawImage();
      };
      image.src = `/media?path=${encodeURIComponent(path)}`;
    }

    function redrawImage() {
      if (!currentImage) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(currentImage, 0, 0);
      boxHitAreas = [];
      drawBoxes(currentBoxes, currentImage.naturalWidth, currentImage.naturalHeight);
      if (draftBox) drawBoxes([draftBox], currentImage.naturalWidth, currentImage.naturalHeight, true);
    }

    function resetImageZoom() {
      imageZoom = 1;
      applyImageZoom();
    }

    function updateZoomIndicator() {
      const percent = Math.round(imageZoom * 100);
      zoomIndicator.innerHTML = `${percent}%<span class="zoom-reset-hint">↺ ESC</span>`;
      zoomIndicator.hidden = percent === 100;
    }

    function applyImageZoom() {
      if (!currentImage) {
        canvas.style.width = "";
        canvas.style.height = "";
        updateZoomIndicator();
        return;
      }
      const wrapper = canvas.parentElement;
      const availableWidth = Math.max(1, wrapper.clientWidth - 32);
      const availableHeight = Math.max(1, wrapper.clientHeight - 32);
      const fitScale = Math.min(
        1,
        availableWidth / currentImage.naturalWidth,
        availableHeight / currentImage.naturalHeight,
      );
      canvas.style.width = `${Math.max(1, Math.round(currentImage.naturalWidth * fitScale * imageZoom))}px`;
      canvas.style.height = `${Math.max(1, Math.round(currentImage.naturalHeight * fitScale * imageZoom))}px`;
      updateZoomIndicator();
    }

    function setImageZoom(nextZoom, clientX, clientY) {
      if (!currentImage) return;
      const wrapper = canvas.parentElement;
      const before = canvas.getBoundingClientRect();
      const ratioX = before.width ? (clientX - before.left) / before.width : 0.5;
      const ratioY = before.height ? (clientY - before.top) / before.height : 0.5;
      imageZoom = Math.min(Math.max(nextZoom, 0.1), 8);
      applyImageZoom();
      const after = canvas.getBoundingClientRect();
      wrapper.scrollLeft += ratioX * (after.width - before.width);
      wrapper.scrollTop += ratioY * (after.height - before.height);
    }

    function drawHistogram(image) {
      const sampleWidth = Math.min(256, image.naturalWidth);
      const sampleHeight = Math.max(1, Math.round(image.naturalHeight * sampleWidth / image.naturalWidth));
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = sampleWidth;
      sampleCanvas.height = sampleHeight;
      const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });
      sampleCtx.drawImage(image, 0, 0, sampleWidth, sampleHeight);
      const pixels = sampleCtx.getImageData(0, 0, sampleWidth, sampleHeight).data;
      const red = new Array(256).fill(0);
      const green = new Array(256).fill(0);
      const blue = new Array(256).fill(0);
      for (let index = 0; index < pixels.length; index += 4) {
        red[pixels[index]] += 1;
        green[pixels[index + 1]] += 1;
        blue[pixels[index + 2]] += 1;
      }
      const width = histogramCanvas.clientWidth || 180;
      const height = histogramCanvas.clientHeight || 92;
      const dpr = window.devicePixelRatio || 1;
      histogramCanvas.width = Math.round(width * dpr);
      histogramCanvas.height = Math.round(height * dpr);
      histogramCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      histogramCtx.clearRect(0, 0, width, height);
      histogramCtx.fillStyle = "#fff";
      histogramCtx.fillRect(0, 0, width, height);
      drawHistogramChannel(red, "#e11d48", width, height);
      drawHistogramChannel(green, "#16a34a", width, height);
      drawHistogramChannel(blue, "#2563eb", width, height);
    }

    function drawHistogramChannel(values, color, width, height) {
      const maxValue = Math.max(...values);
      if (!maxValue) return;
      histogramCtx.beginPath();
      values.forEach((value, index) => {
        const x = index / 255 * width;
        const y = height - value / maxValue * (height - 8) - 4;
        if (index === 0) histogramCtx.moveTo(x, y);
        else histogramCtx.lineTo(x, y);
      });
      histogramCtx.strokeStyle = color;
      histogramCtx.globalAlpha = 0.82;
      histogramCtx.lineWidth = 1.4;
      histogramCtx.stroke();
      histogramCtx.globalAlpha = 1;
    }

    async function loadExif(path) {
      exifEl.innerHTML = '<div class="empty">读取中...</div>';
      const data = await getJson(`/api/exif?path=${encodeURIComponent(path)}`);
      const rows = [];
      Object.entries(data.info).forEach(([key, value]) => rows.push([key, value]));
      Object.entries(data.exif).forEach(([key, value]) => rows.push([key, value]));
      if (!rows.length) {
        exifEl.innerHTML = '<div class="empty">没有 EXIF 信息</div>';
        return;
      }
      exifEl.innerHTML = "";
      rows.forEach(([key, value]) => {
        const row = document.createElement("div");
        row.className = "kv-row";
        row.innerHTML = `<div class="kv-key"></div><div class="kv-value"></div>`;
        row.children[0].textContent = key;
        row.children[1].textContent = Array.isArray(value) ? value.join(", ") : String(value);
        exifEl.appendChild(row);
      });
    }

    function boxEdges(box) {
      return {
        left: box.cx - box.width / 2,
        right: box.cx + box.width / 2,
        top: box.cy - box.height / 2,
        bottom: box.cy + box.height / 2,
      };
    }

    function normalizeBoxEdges(edges, template) {
      const left = Math.min(Math.max(edges.left, 0), 1);
      const right = Math.min(Math.max(edges.right, 0), 1);
      const top = Math.min(Math.max(edges.top, 0), 1);
      const bottom = Math.min(Math.max(edges.bottom, 0), 1);
      return {
        ...template,
        cx: (left + right) / 2,
        cy: (top + bottom) / 2,
        width: Math.max(0.003, right - left),
        height: Math.max(0.003, bottom - top),
      };
    }

    function drawSelectedAnchors(box, imageWidth, imageHeight, color) {
      const edges = boxEdges(box);
      const points = [
        [edges.left, edges.top],
        [box.cx, edges.top],
        [edges.right, edges.top],
        [edges.right, box.cy],
        [edges.right, edges.bottom],
        [box.cx, edges.bottom],
        [edges.left, edges.bottom],
        [edges.left, box.cy],
      ];
      const size = Math.max(7, Math.round(Math.min(imageWidth, imageHeight) / 120));
      ctx.save();
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [x, y] of points) {
        ctx.fillRect(x * imageWidth - size / 2, y * imageHeight - size / 2, size, size);
        ctx.strokeRect(x * imageWidth - size / 2, y * imageHeight - size / 2, size, size);
      }
      ctx.restore();
    }

    function anchorHitAreas(box, imageWidth, imageHeight) {
      const edges = boxEdges(box);
      const size = Math.max(10, Math.round(Math.min(imageWidth, imageHeight) / 95));
      const halfX = size / imageWidth / 2;
      const halfY = size / imageHeight / 2;
      return [
        ["nw", edges.left, edges.top],
        ["n", box.cx, edges.top],
        ["ne", edges.right, edges.top],
        ["e", edges.right, box.cy],
        ["se", edges.right, edges.bottom],
        ["s", box.cx, edges.bottom],
        ["sw", edges.left, edges.bottom],
        ["w", edges.left, box.cy],
      ].map(([name, x, y]) => ({
        name,
        left: x - halfX,
        right: x + halfX,
        top: y - halfY,
        bottom: y + halfY,
      }));
    }

    function drawBoxes(boxes, imageWidth, imageHeight, draft = false) {
      ctx.lineWidth = Math.max(2, Math.round(Math.min(imageWidth, imageHeight) / 360));
      ctx.font = `${Math.max(14, Math.round(imageWidth / 80))}px sans-serif`;
      boxes.forEach((box, index) => {
        const color = colors[Math.abs(box.class_id) % colors.length];
        const x = (box.cx - box.width / 2) * imageWidth;
        const y = (box.cy - box.height / 2) * imageHeight;
        const width = box.width * imageWidth;
        const height = box.height * imageHeight;
        ctx.strokeStyle = color;
        if (maskEnabled) {
          ctx.save();
          ctx.globalAlpha = draft ? 0.18 : 0.26;
          ctx.fillStyle = color;
          ctx.fillRect(x, y, width, height);
          ctx.restore();
        }
        if (draft) ctx.setLineDash([8, 6]);
        ctx.strokeRect(x, y, width, height);
        ctx.setLineDash([]);
        const text = box.label;
        const metrics = ctx.measureText(text);
        const labelTop = Math.max(0, y - 22);
        const labelWidth = metrics.width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(x, labelTop, labelWidth, 22);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x + 5, Math.max(16, y - 6));
        if (!draft) {
          boxHitAreas.push({
            index,
            box: {
              left: x / imageWidth,
              top: y / imageHeight,
              right: (x + width) / imageWidth,
              bottom: (y + height) / imageHeight,
            },
            label: {
              left: x / imageWidth,
              top: labelTop / imageHeight,
              right: (x + labelWidth) / imageWidth,
              bottom: (labelTop + 22) / imageHeight,
            },
            anchors: anchorHitAreas(box, imageWidth, imageHeight),
          });
          if (index === selectedBoxIndex) drawSelectedAnchors(box, imageWidth, imageHeight, color);
        }
      });
    }

    function cloneBoxes(boxes) {
      return boxes.map(box => ({...box}));
    }

    function setBoxesDirty(dirty) {
      boxesDirty = dirty;
      updateAnnotationButtons();
    }

    function defaultBoxLabel() {
      return selectedClassLabel || classLabels[selectedClassId] || String(selectedClassId);
    }

    function selectClassLabel(classId, label, row) {
      selectedClassId = classId;
      selectedClassLabel = label;
      document.querySelectorAll(".label-row").forEach(item => item.classList.remove("active"));
      if (row) row.classList.add("active");
    }

    function updateAnnotationButtons() {
      editModeToggle.classList.toggle("active", annotateMode);
      editModeToggle.querySelector(".header-icon").textContent = annotateMode ? "●" : "◌";
      editModeToggle.querySelector("span:last-child").textContent = annotateMode ? "打标" : "只读";
      editModeToggle.title = annotateMode ? "当前为打标状态" : "当前为只读状态";
      canvas.classList.toggle("annotating", annotateMode);
      const locked = Boolean(lockedByOther);
      saveAnnotation.disabled = locked || !boxesDirty;
      resetAnnotation.disabled = locked || !boxesDirty;
      deleteAnnotation.disabled = locked || !currentBoxes.length;
      editModeToggle.disabled = locked;
      if (locked) {
        editModeToggle.title = `团队模式：${lockedByOther.username} 正在标注该图片`;
      }
    }

    async function loadLabels() {
      const data = await getJson("/api/classes");
      classLabels = data.classes;
      labelsEl.innerHTML = "";
      const groups = data.class_groups?.length
        ? data.class_groups
        : data.classes_file
          ? [{classes_file: data.classes_file, classes: data.classes}]
          : [];
      let selectedAssigned = false;
      groups.forEach((group, groupIndex) => {
        const source = document.createElement("div");
        source.className = "label-source";
        source.textContent = group.classes_file;
        labelsEl.appendChild(source);
        group.classes.forEach((label, index) => {
          const row = document.createElement("button");
          row.type = "button";
          row.className = "label-row";
          row.innerHTML = `<span class="swatch" style="background:${colors[index % colors.length]}"></span><span>${index}: ${escapeHtml(label)}</span>`;
          row.onclick = () => selectClassLabel(index, label, row);
          labelsEl.appendChild(row);
          if (!selectedAssigned) {
            selectClassLabel(index, label, row);
            selectedAssigned = true;
          }
        });
      });
      if (!groups.length) {
        selectedClassId = 0;
        selectedClassLabel = "0";
        labelsEl.innerHTML = '<div class="empty">未找到 classes.txt</div>';
      }
    }

    async function loadStatistics() {
      statisticsData = await getJson("/api/statistics");
      renderStatistics();
    }

    async function loadConfig() {
      const embeddedTeamMode = document.body.dataset.teamMode === "true";
      try {
        const data = await getJson("/api/config");
        teamMode = Boolean(data.team_mode);
        shareUrl = data.share_url || window.location.origin;
      } catch (error) {
        teamMode = embeddedTeamMode;
      }
      document.body.classList.toggle("team-mode", teamMode);
      if (!teamMode) {
        document.body.classList.remove("username-required");
        username = "";
        usernameReady = Promise.resolve();
        onlineCount.textContent = "-";
        collaborationUsers.innerHTML = "";
        return;
      }
      usernameReady = requireUsername();
      await usernameReady;
      username = window.yoloutilsUsername || usernameInput.value.trim();
      await updatePresence();
    }

    function requireUsername() {
      bindUsernameGate();
      username = window.yoloutilsUsername || usernameInput.value.trim();
      if (username) return Promise.resolve();
      document.body.classList.add("username-required");
      setTimeout(() => usernameInput.focus(), 0);
      return usernameReady;
    }

    function submitUsername(event = null) {
      window.yoloutilsSubmitUsername?.(event);
      username = window.yoloutilsUsername || usernameInput.value.trim();
      if (teamMode && username) updatePresence();
    }

    function bindUsernameGate() {
      if (usernameGateBound) return;
      usernameGateBound = true;
      usernameForm.addEventListener("submit", submitUsername);
      usernameSubmit.addEventListener("click", submitUsername);
      usernameInput.addEventListener("keydown", event => {
        if (event.key === "Enter") submitUsername(event);
      });
    }

    async function loadConsoleLogs() {
      try {
        const data = await getJson("/api/logs?lines=200");
        consoleLog.textContent = [`日志文件: ${data.file}`, "", ...data.lines].join("\\n");
      } catch (error) {
        consoleLog.textContent = `读取日志失败: ${error.message}`;
      }
    }

    async function toggleConsole() {
      document.body.classList.toggle("console-open");
      const button = statsEl.querySelector("#consoleToggle");
      if (button) button.classList.toggle("active", document.body.classList.contains("console-open"));
      if (document.body.classList.contains("console-open")) await loadConsoleLogs();
    }

    function clientId() {
      const key = "yoloutils-workstation-client-id";
      let id = "";
      try {
        id = localStorage.getItem(key);
      } catch (error) {
        id = "";
      }
      if (!id) {
        id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
        try {
          localStorage.setItem(key, id);
        } catch (error) {
          window.yoloutilsClientId = id;
        }
      }
      return id || window.yoloutilsClientId;
    }

    async function updatePresence() {
      if (!teamMode) return;
      await usernameReady;
      try {
        const data = await postJson("/api/presence", {client_id: clientId(), username: username || "独立用户"});
        onlineCount.textContent = data.online;
        collaborationUsers.innerHTML = data.users
          .map(user => `<div class="collaboration-user">${escapeHtml(user.username)}</div>`)
          .join("");
      } catch (error) {
        onlineCount.textContent = "-";
      }
    }

    function leavePresence() {
      if (!teamMode) return;
      const payload = JSON.stringify({client_id: clientId()});
      if (navigator.sendBeacon) {
        navigator.sendBeacon("/api/presence/leave", new Blob([payload], {type: "application/json"}));
        return;
      }
      fetch("/api/presence/leave", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: payload,
        keepalive: true,
      }).catch(() => {});
    }

    async function acquireCurrentLock() {
      if (!teamMode || !currentPath) return;
      await usernameReady;
      const data = await postJson("/api/lock", {
        path: currentPath,
        client_id: clientId(),
        username,
      });
      lockedByOther = data.locked ? data.owner : null;
      if (lockedByOther) {
        annotateMode = false;
        viewerTitle.innerHTML = `<span class="pane-title-icon">▧</span>${escapeHtml(currentPath)} <span class="lock-note">锁定：${escapeHtml(lockedByOther.username)}</span>`;
      }
      updateAnnotationButtons();
    }

    async function releaseCurrentLock(path = currentPath) {
      if (!teamMode || !path) return;
      try {
        await postJson("/api/lock/release", {path, client_id: clientId()});
      } catch (error) {
        // Ignore lock release errors; stale locks expire server-side.
      }
    }

    function renderBreadcrumb() {
      const container = document.createElement("div");
      container.className = "breadcrumb";
      const home = document.createElement("span");
      home.className = "breadcrumb-home";
      home.textContent = "⌂";
      home.title = "Home";
      container.appendChild(home);
      const currentPathParts = currentPath ? currentPath.split("/").filter(Boolean) : [];
      const fileName = currentPathParts.length ? currentPathParts[currentPathParts.length - 1] : "";
      const parts = currentPathParts.length
        ? currentPathParts.slice(0, -1)
        : currentDir ? currentDir.split("/").filter(Boolean) : [];
      const crumbs = [];
      let path = "";
      for (const part of parts) {
        path = path ? `${path}/${part}` : part;
        crumbs.push({name: part, path});
      }
      container.title = currentPath || currentDir || ".";
      crumbs.forEach((crumb, index) => {
        if (index > 0) {
          const separator = document.createElement("span");
          separator.className = "breadcrumb-separator";
          separator.textContent = ">";
          container.appendChild(separator);
        }
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = crumb.name;
        button.title = crumb.path || ".";
        button.onclick = () => selectDir(crumb.path);
        container.appendChild(button);
      });
      if (fileName) {
        if (crumbs.length) {
          const separator = document.createElement("span");
          separator.className = "breadcrumb-separator";
          separator.textContent = ">";
          container.appendChild(separator);
        }
        const file = document.createElement("span");
        file.className = "breadcrumb-file";
        file.textContent = fileName;
        file.title = currentPath;
        container.appendChild(file);
      }
      return container;
    }

    function renderStatistics() {
      if (!statisticsData) return;
      const data = statisticsData;
      statsEl.innerHTML = `
        <div class="stats-left"></div>
        <div class="stats-right">
          <span class="stat"><span class="stat-icon">▧</span>图像 <strong>${data.images}</strong></span>
          <span class="stat"><span class="stat-icon">✓</span>已完成 <strong>${data.txt_valid}/${data.images}</strong></span>
          <span class="stat"><span class="stat-icon">▯</span>.txt <strong>${data.txt_total}</strong></span>
          <span class="stat"><span class="stat-icon">⌑</span>classes.txt <strong>${data.classes_files}</strong></span>
          <span class="stat bad"><span class="stat-icon">✕</span>损坏图像 <strong>${data.images_damaged}</strong></span>
          <span class="stat bad"><span class="stat-icon">!</span>无效 .txt <strong>${data.txt_invalid_total}</strong></span>
          <button id="consoleToggle" class="footer-button" type="button"><span class="stat-icon">▤</span>控制台</button>
        </div>
      `;
      statsEl.querySelector(".stats-left").appendChild(renderBreadcrumb());
      const button = statsEl.querySelector("#consoleToggle");
      button.classList.toggle("active", document.body.classList.contains("console-open"));
      button.addEventListener("click", toggleConsole);
    }

    async function init() {
      bindUsernameGate();
      canvas.style.display = "none";
      await loadConfig();
      await loadTree();
      await loadLabels();
      await loadStatistics();
      const rootButton = document.querySelector("#tree .tree-select");
      if (rootButton) await selectDir("", rootButton);
      initSplitter();
      initHistogramSplitter();
      initLeftPanel();
      initMainSplitter();
      initRightPanel();
      initViewerTools();
      initViewerFocus();
      initHeaderActions();
      initFileSorting();
      initImageZoom();
      initKeyboardShortcuts();
      initCanvasAnnotation();
      initPresence();
      updateAnnotationButtons();
    }

    function initPresence() {
      updatePresence();
      setInterval(updatePresence, 5000);
      setInterval(() => {
        if (document.body.classList.contains("console-open")) loadConsoleLogs();
      }, 5000);
      window.addEventListener("pagehide", leavePresence);
      window.addEventListener("beforeunload", () => {
        leavePresence();
        if (teamMode && currentPath) {
          navigator.sendBeacon?.(
            "/api/lock/release",
            new Blob(
              [JSON.stringify({path: currentPath, client_id: clientId()})],
              {type: "application/json"},
            ),
          );
        }
      });
    }

    function initFileSorting() {
      toggleFileSort.addEventListener("click", () => {
        fileSortDirection = fileSortDirection === "asc" ? "desc" : "asc";
        renderFiles();
      });
    }

    function initHeaderActions() {
      annotateModeButton.addEventListener("click", () => alert("当前窗口就是标注窗口"));
      editModeToggle.addEventListener("click", () => {
        if (lockedByOther) return;
        annotateMode = !annotateMode;
        updateAnnotationButtons();
      });
      shareButton.addEventListener("click", shareCurrentLocation);
      downloadImage.addEventListener("click", downloadCurrentImage);
      queryButton.addEventListener("click", queryLocation);
      shortcutButton.addEventListener("click", () => {
        shortcutPopover.hidden = !shortcutPopover.hidden;
      });
      closeShortcuts.addEventListener("click", () => {
        shortcutPopover.hidden = true;
      });
      datasetButton.addEventListener("click", () => alert("数据集功能入口已预留"));
      trainButton.addEventListener("click", () => alert("训练功能入口已预留"));
    }

    function visibleTreeButtons() {
      return Array.from(document.querySelectorAll("#tree .tree-select"))
        .filter(button => button.offsetParent !== null);
    }

    async function selectAdjacentTree(direction) {
      const buttons = visibleTreeButtons();
      if (!buttons.length) return;
      const currentIndex = buttons.findIndex(button => button.dataset.path === currentDir);
      const index = currentIndex < 0
        ? 0
        : Math.min(Math.max(currentIndex + direction, 0), buttons.length - 1);
      const button = buttons[index];
      await selectDir(button.dataset.path || "", button);
      button.scrollIntoView({block: "nearest"});
    }

    function downloadCurrentImage() {
      if (!currentPath) {
        alert("请选择图片");
        return;
      }
      const link = document.createElement("a");
      link.href = `/media?path=${encodeURIComponent(currentPath)}`;
      link.download = currentPath.split("/").pop() || "image";
      document.body.appendChild(link);
      link.click();
      link.remove();
    }

    async function shareCurrentLocation() {
      const serverUrl = shareUrl || window.location.origin;
      const payload = {
        title: "Yolo Workstation",
        text: serverUrl,
        url: serverUrl,
      };
      try {
        if (navigator.share) {
          await navigator.share(payload);
          return;
        }
        await navigator.clipboard.writeText(serverUrl);
        alert("已复制服务地址");
      } catch (error) {
        if (navigator.clipboard) {
          await navigator.clipboard.writeText(serverUrl);
          alert("已复制服务地址");
        }
      }
    }

    async function queryLocation() {
      const keyword = prompt("查询目录或当前文件", currentPath || currentDir || "");
      if (!keyword || !keyword.trim()) return;
      const query = keyword.trim().toLowerCase();
      const dirButton = Array.from(document.querySelectorAll("#tree button")).find(button => {
        const path = (button.dataset.path || "").toLowerCase();
        const name = button.querySelector(".tree-name")?.textContent.toLowerCase() || "";
        return path === query || path.includes(query) || name === query || name.includes(query);
      });
      if (dirButton) {
        await selectDir(dirButton.dataset.path || "", dirButton);
        return;
      }
      const file = currentFiles.find(item => {
        const name = item.name.toLowerCase();
        const path = item.path.toLowerCase();
        return name === query || path === query || name.includes(query) || path.includes(query);
      });
      if (file) {
        const row = Array.from(document.querySelectorAll("#files button"))
          .find(button => button.dataset.path === file.path);
        await selectImage(file.path, row);
        return;
      }
      alert("没有找到匹配的目录或文件");
    }

    function initViewerFocus() {
      viewerHeader.addEventListener("dblclick", event => {
        if (event.target.closest("button")) return;
        appMain.classList.toggle("viewer-focus");
        applyImageZoom();
        redrawHistogram();
      });
    }

    function initImageZoom() {
      canvas.parentElement.addEventListener("wheel", event => {
        if (!currentImage) return;
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
        setImageZoom(imageZoom * factor, event.clientX, event.clientY);
      }, {passive: false});
      window.addEventListener("resize", applyImageZoom);
    }

    function initKeyboardShortcuts() {
      window.addEventListener("keydown", event => {
        if (event.target instanceof Element && event.target.closest("input, textarea, select")) return;
        if (event.key === "Escape") {
          if (!shortcutPopover.hidden) {
            shortcutPopover.hidden = true;
          } else if (currentImage) {
            resetImageZoom();
          }
          return;
        }
        if (!event.metaKey && !event.ctrlKey && event.key === "ArrowUp") {
          event.preventDefault();
          selectAdjacentTree(-1);
          return;
        }
        if (!event.metaKey && !event.ctrlKey && event.key === "ArrowDown") {
          event.preventDefault();
          selectAdjacentTree(1);
          return;
        }
        if (!event.metaKey && !event.ctrlKey && event.key === "ArrowLeft") {
          event.preventDefault();
          selectAdjacentImage(-1);
          return;
        }
        if (!event.metaKey && !event.ctrlKey && event.key === "ArrowRight") {
          event.preventDefault();
          selectAdjacentImage(1);
          return;
        }
        if (!(event.metaKey || event.ctrlKey)) return;
        const key = event.key.toLowerCase();
        if (key === "d") {
          event.preventDefault();
          if (!deleteAnnotation.disabled) deleteAnnotation.click();
        } else if (key === "m") {
          event.preventDefault();
          maskAnnotation.click();
        } else if (key === "r") {
          event.preventDefault();
          if (!resetAnnotation.disabled) resetAnnotation.click();
        } else if (key === "s") {
          event.preventDefault();
          if (!saveAnnotation.disabled) saveAnnotation.click();
        }
      });
    }

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
        y: Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1),
      };
    }

    function rectContains(rect, point) {
      return point.x >= rect.left && point.x <= rect.right && point.y >= rect.top && point.y <= rect.bottom;
    }

    function hitTestBox(point) {
      for (let areaIndex = boxHitAreas.length - 1; areaIndex >= 0; areaIndex -= 1) {
        const area = boxHitAreas[areaIndex];
        for (const anchor of area.anchors) {
          if (rectContains(anchor, point)) return {type: "anchor", index: area.index, anchor: anchor.name};
        }
        if (rectContains(area.label, point)) return {type: "label", index: area.index};
        if (rectContains(area.box, point)) return {type: "box", index: area.index};
      }
      return null;
    }

    function moveBox(box, dx, dy) {
      const halfWidth = box.width / 2;
      const halfHeight = box.height / 2;
      return {
        ...box,
        cx: Math.min(Math.max(box.cx + dx, halfWidth), 1 - halfWidth),
        cy: Math.min(Math.max(box.cy + dy, halfHeight), 1 - halfHeight),
      };
    }

    function resizeBox(box, anchor, point) {
      const edges = boxEdges(box);
      const minSize = 0.003;
      if (anchor.includes("w")) edges.left = Math.min(point.x, edges.right - minSize);
      if (anchor.includes("e")) edges.right = Math.max(point.x, edges.left + minSize);
      if (anchor.includes("n")) edges.top = Math.min(point.y, edges.bottom - minSize);
      if (anchor.includes("s")) edges.bottom = Math.max(point.y, edges.top + minSize);
      edges.left = Math.max(0, edges.left);
      edges.right = Math.min(1, edges.right);
      edges.top = Math.max(0, edges.top);
      edges.bottom = Math.min(1, edges.bottom);
      return normalizeBoxEdges(edges, box);
    }

    function boxFromPoints(start, end) {
      const left = Math.min(start.x, end.x);
      const right = Math.max(start.x, end.x);
      const top = Math.min(start.y, end.y);
      const bottom = Math.max(start.y, end.y);
      return {
        class_id: selectedClassId,
        label: defaultBoxLabel(),
        cx: (left + right) / 2,
        cy: (top + bottom) / 2,
        width: right - left,
        height: bottom - top,
      };
    }

    function initCanvasAnnotation() {
      canvas.addEventListener("mousedown", event => {
        if (!currentImage || event.button !== 0 || lockedByOther) return;
        const point = canvasPoint(event);
        const hit = hitTestBox(point);
        if (hit) {
          selectedBoxIndex = hit.index;
          draftBox = null;
          dragStart = null;
          if (hit.type === "anchor") {
            canvasInteraction = {
              type: "resize",
              index: hit.index,
              anchor: hit.anchor,
              startBox: {...currentBoxes[hit.index]},
            };
          } else if (hit.type === "label") {
            canvasInteraction = {
              type: "move",
              index: hit.index,
              startPoint: point,
              startBox: {...currentBoxes[hit.index]},
            };
          } else {
            canvasInteraction = {type: "select"};
          }
          redrawImage();
          event.preventDefault();
          return;
        }
        selectedBoxIndex = -1;
        if (annotateMode) {
          dragStart = point;
          draftBox = null;
          canvasInteraction = {type: "create"};
        } else {
          const wrapper = canvas.parentElement;
          canvasInteraction = {
            type: "pan",
            startClientX: event.clientX,
            startClientY: event.clientY,
            scrollLeft: wrapper.scrollLeft,
            scrollTop: wrapper.scrollTop,
          };
          canvas.classList.add("panning");
        }
        redrawImage();
        event.preventDefault();
      });
      window.addEventListener("mousemove", event => {
        if (!canvasInteraction) return;
        if (canvasInteraction.type === "pan") {
          const wrapper = canvas.parentElement;
          wrapper.scrollLeft = canvasInteraction.scrollLeft - (event.clientX - canvasInteraction.startClientX);
          wrapper.scrollTop = canvasInteraction.scrollTop - (event.clientY - canvasInteraction.startClientY);
          return;
        }
        const point = canvasPoint(event);
        if (canvasInteraction.type === "create" && dragStart) {
          draftBox = boxFromPoints(dragStart, point);
        } else if (canvasInteraction.type === "move") {
          const dx = point.x - canvasInteraction.startPoint.x;
          const dy = point.y - canvasInteraction.startPoint.y;
          currentBoxes[canvasInteraction.index] = moveBox(canvasInteraction.startBox, dx, dy);
        } else if (canvasInteraction.type === "resize") {
          currentBoxes[canvasInteraction.index] = resizeBox(
            canvasInteraction.startBox,
            canvasInteraction.anchor,
            point,
          );
        }
        redrawImage();
      });
      window.addEventListener("mouseup", event => {
        if (!canvasInteraction) return;
        if (canvasInteraction.type === "create" && dragStart) {
          const point = canvasPoint(event);
          const box = boxFromPoints(dragStart, point);
          if (box.width >= 0.003 && box.height >= 0.003) {
            currentBoxes.push(box);
            selectedBoxIndex = currentBoxes.length - 1;
            setBoxesDirty(true);
          }
        } else if (canvasInteraction.type === "move" || canvasInteraction.type === "resize") {
          setBoxesDirty(true);
        }
        canvasInteraction = null;
        dragStart = null;
        draftBox = null;
        canvas.classList.remove("panning");
        redrawImage();
      });
    }

    function initViewerTools() {
      autoAnnotate.addEventListener("click", () => {
        autoAnnotate.classList.toggle("active");
      });
      maskAnnotation.addEventListener("click", () => {
        maskEnabled = !maskEnabled;
        maskAnnotation.classList.toggle("active", maskEnabled);
        redrawImage();
      });
      deleteAnnotation.addEventListener("click", () => {
        if (!currentPath || !currentBoxes.length) return;
        currentBoxes = [];
        selectedBoxIndex = -1;
        setBoxesDirty(true);
        redrawImage();
      });
      resetAnnotation.addEventListener("click", () => {
        if (!currentPath) return;
        currentBoxes = cloneBoxes(savedBoxes);
        draftBox = null;
        dragStart = null;
        selectedBoxIndex = -1;
        canvasInteraction = null;
        setBoxesDirty(false);
        redrawImage();
      });
      saveAnnotation.addEventListener("click", async () => {
        if (!currentPath || !boxesDirty) return;
        const savedPath = currentPath;
        let annotation;
        try {
          annotation = await postJson("/api/annotation", {
            path: currentPath,
            client_id: clientId(),
            username: username || "独立用户",
            boxes: currentBoxes,
          });
        } catch (error) {
          alert(error.message);
          await acquireCurrentLock();
          return;
        }
        lockedByOther = null;
        currentBoxes = cloneBoxes(annotation.boxes);
        savedBoxes = cloneBoxes(annotation.boxes);
        setBoxesDirty(false);
        redrawImage();
        await loadStatistics();
        await loadTree();
        await refreshCurrentFiles();
        const nextPath = nextAnnotationFilePath(savedPath);
        if (nextPath) {
          await selectImage(nextPath, fileRow(nextPath));
        }
      });
    }

    function initSplitter() {
      let dragging = false;
      splitter.addEventListener("mousedown", event => {
        dragging = true;
        splitter.classList.add("dragging");
        event.preventDefault();
      });
      window.addEventListener("mousemove", event => {
        if (!dragging || rightPanel.classList.contains("exif-collapsed")) return;
        const rect = rightPanel.getBoundingClientRect();
        const min = 96;
        const splitterSize = splitter.getBoundingClientRect().height;
        const topHeight = Math.min(Math.max(event.clientY - rect.top, min), rect.height - min - splitterSize);
        topInfoPane.style.flexBasis = `${topHeight}px`;
        topInfoPane.style.flexGrow = "0";
        redrawHistogram();
      });
      window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        splitter.classList.remove("dragging");
      });
    }

    function initHistogramSplitter() {
      let dragging = false;
      histogramSplitter.addEventListener("mousedown", event => {
        dragging = true;
        histogramSplitter.classList.add("dragging");
        event.preventDefault();
      });
      window.addEventListener("mousemove", event => {
        if (!dragging || rightPanel.classList.contains("histogram-collapsed")) return;
        const rect = topInfoPane.getBoundingClientRect();
        const min = 42;
        const splitterSize = histogramSplitter.getBoundingClientRect().height;
        const maxLabels = rect.height - min - splitterSize;
        const boundedMaxLabels = Math.max(min, maxLabels);
        const labelsHeight = Math.min(Math.max(event.clientY - rect.top, min), boundedMaxLabels);
        labelsPane.style.flexBasis = `${labelsHeight}px`;
        labelsPane.style.flexGrow = "0";
        histogramPane.style.flexBasis = `${Math.max(min, rect.height - labelsHeight - splitterSize)}px`;
        histogramPane.style.flexGrow = "1";
        redrawHistogram();
      });
      window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        histogramSplitter.classList.remove("dragging");
        redrawHistogram();
      });
    }

    function redrawHistogram() {
      if (!currentImage || rightPanel.classList.contains("histogram-collapsed")) return;
      requestAnimationFrame(() => drawHistogram(currentImage));
    }

    function initLeftPanel() {
      let dragging = false;
      reloadTree.addEventListener("click", async () => {
        await loadTree();
        await loadStatistics();
        await selectDir(currentDir);
      });
      leftSplitter.addEventListener("mousedown", event => {
        dragging = true;
        leftSplitter.classList.add("dragging");
        event.preventDefault();
      });
      window.addEventListener("mousemove", event => {
        if (!dragging || leftPanel.classList.contains("tree-hidden")) return;
        const rect = leftPanel.getBoundingClientRect();
        const minTree = 120;
        const minFiles = 160;
        const splitterWidth = leftSplitter.getBoundingClientRect().width;
        const treeWidth = Math.min(Math.max(event.clientX - rect.left, minTree), rect.width - minFiles - splitterWidth);
        leftPanel.style.gridTemplateColumns = `${treeWidth}px 8px minmax(${minFiles}px, 1fr)`;
      });
      window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        leftSplitter.classList.remove("dragging");
      });
      toggleTree.addEventListener("click", () => {
        leftPanel.classList.add("tree-hidden");
        appMain.classList.add("tree-hidden");
        setMainColumns(280);
      });
      showTree.addEventListener("click", () => {
        leftPanel.classList.remove("tree-hidden");
        appMain.classList.remove("tree-hidden");
        if (!leftPanel.style.gridTemplateColumns || leftPanel.style.gridTemplateColumns.startsWith("0px")) {
          leftPanel.style.gridTemplateColumns = "minmax(120px, 46%) 8px minmax(160px, 1fr)";
        }
        setMainColumns(520);
      });
    }

    function setMainColumns(leftWidth) {
      const rightWidth = appMain.classList.contains("right-hidden") ? 0 : 220;
      appMain.style.gridTemplateColumns = `${leftWidth}px 8px minmax(420px, 1fr) ${rightWidth}px`;
    }

    function initMainSplitter() {
      let dragging = false;
      mainSplitter.addEventListener("mousedown", event => {
        dragging = true;
        mainSplitter.classList.add("dragging");
        event.preventDefault();
      });
      window.addEventListener("mousemove", event => {
        if (!dragging) return;
        const rect = appMain.getBoundingClientRect();
        const minLeft = appMain.classList.contains("tree-hidden") ? 220 : 360;
        const minViewer = 420;
        const splitterWidth = mainSplitter.getBoundingClientRect().width;
        const rightWidth = appMain.classList.contains("right-hidden") ? 0 : 220;
        const maxLeft = rect.width - minViewer - splitterWidth - rightWidth;
        const boundedMaxLeft = Math.max(minLeft, maxLeft);
        const leftWidth = Math.min(Math.max(event.clientX - rect.left, minLeft), boundedMaxLeft);
        setMainColumns(leftWidth);
      });
      window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        mainSplitter.classList.remove("dragging");
      });
    }

    function initRightPanel() {
      hideRight.addEventListener("click", () => {
        appMain.classList.add("right-hidden");
        const leftWidth = leftPanel.getBoundingClientRect().width;
        setMainColumns(leftWidth);
      });
      showRight.addEventListener("click", () => {
        appMain.classList.remove("right-hidden");
        const leftWidth = leftPanel.getBoundingClientRect().width;
        setMainColumns(leftWidth);
      });
      toggleHistogram.addEventListener("click", () => {
        const isCollapsed = rightPanel.classList.contains("histogram-collapsed");
        if (isCollapsed) {
          rightPanel.classList.remove("histogram-collapsed");
          if (histogramExpandedTopHeight) {
            topInfoPane.style.flexBasis = `${histogramExpandedTopHeight}px`;
            topInfoPane.style.flexGrow = "0";
          }
          toggleHistogram.textContent = "⌄";
        } else {
          const topHeight = topInfoPane.getBoundingClientRect().height;
          const labelsHeight = labelsPane.getBoundingClientRect().height;
          const histogramHeaderHeight = histogramPane.querySelector(".pane-header").getBoundingClientRect().height;
          histogramExpandedTopHeight = topHeight;
          labelsPane.style.flexBasis = `${labelsHeight}px`;
          labelsPane.style.flexGrow = "0";
          topInfoPane.style.flexBasis = `${labelsHeight + histogramHeaderHeight}px`;
          topInfoPane.style.flexGrow = "0";
          rightPanel.classList.add("histogram-collapsed");
          toggleHistogram.textContent = "⌃";
        }
        redrawHistogram();
      });
      toggleExif.addEventListener("click", () => {
        rightPanel.classList.toggle("exif-collapsed");
        toggleExif.textContent = rightPanel.classList.contains("exif-collapsed") ? "⌃" : "⌄";
      });
    }
    init().catch(error => {
      document.body.innerHTML = `<pre>${error.stack || error}</pre>`;
    });
  </script>
</body>
</html>""".replace(
            "__BODY_CLASS__",
            "team-mode username-required" if self.team_mode else "",
        ).replace("__TEAM_MODE__", "true" if self.team_mode else "false")

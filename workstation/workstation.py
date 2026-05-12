import os
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from io import BytesIO
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from PIL import ExifTags, Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pillow_heif = None
except ImportError:
    FastAPI = None
    HTTPException = None
    Query = None
    Request = None
    FileResponse = None
    HTMLResponse = None
    Response = None
    StaticFiles = None
    uvicorn = None
    ExifTags = None
    Image = None
    pillow_heif = None

class Common:
    image_exts = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
    )


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
        self.mdns = "netkiller.local"
        self.presence = {}
        self.locks = {}
        self.log_root = None
        self.classes_file = None
        self.class_groups = []
        self.classes = []
        self.model_root = None
        self.model_cache = {}
        self.statistics_cache = None
        self.statistics_cache_at = 0.0

    def main(
        self,
        workspace: str,
        dataset: str = None,
        run: str = None,
        classes_file: str = None,
        open_browser: bool = False,
        team_mode: bool = False,
        mdns: str = "netkiller.local",
    ):
        if FastAPI is None or uvicorn is None:
            print("缺少依赖: fastapi/uvicorn，请先安装: pip install fastapi uvicorn")
            return

        self.workspace = Path(workspace).expanduser().resolve()
        self.log_root = self.workspace
        if not self.workspace.is_dir():
            print(f"workspace 目录不存在: {self.workspace}")
            return
        self.dataset = Path(dataset).expanduser().resolve() if dataset else None
        self.run = Path(run).expanduser().resolve() if run else None
        self.requested_classes_file = classes_file
        self.open_browser = open_browser
        self.team_mode = team_mode
        self.mdns = self._normalize_mdns(mdns)

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
        mdns = os.environ.get("YOLOUTILS_MDNS", "").strip()
        host = self._normalize_mdns(mdns) if mdns else self._lan_ip_address()
        if not host:
            host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def _lan_ip_address(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                host = sock.getsockname()[0]
                if host and not host.startswith("127."):
                    return host
        except OSError:
            pass
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                host = item[4][0]
                if host and not host.startswith("127."):
                    return host
        except OSError:
            pass
        return ""

    def _normalize_mdns(self, value: str):
        name = (value or "netkiller.local").strip().lower()
        if "://" in name:
            name = name.split("://", 1)[1]
        name = name.split("/", 1)[0].split(":", 1)[0]
        return name if name.endswith(".local") else f"{name}.local"

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
        return (self.log_root or self.workspace) / ".yoloutils-workstation.pid"

    def _log_file(self):
        return (self.log_root or self.workspace) / ".project.log"

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

    def _classes_text(self):
        if not self.class_groups:
            return ""
        try:
            return self.class_groups[0]["path"].read_text(encoding="utf-8")
        except OSError:
            return ""

    def _save_classes_text(self, content: str):
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        if not lines:
            raise HTTPException(status_code=400, detail="classes.txt 不能为空")
        target = self.class_groups[0]["path"] if self.class_groups else self.workspace / "classes.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.class_groups = self._load_class_groups()
        self.classes_file = self.class_groups[0]["path"] if self.class_groups else None
        self.classes = self.class_groups[0]["classes"] if self.class_groups else []
        return target

    def _max_class_count(self):
        return max([len(group["classes"]) for group in self.class_groups] + [len(self.classes), 0])

    def _is_image(self, path: Path):
        return path.is_file() and path.name.lower().endswith(Common.image_exts)

    def _needs_browser_conversion(self, path: Path):
        return path.suffix.lower() in (".heic", ".heif", ".tif", ".tiff")

    def _image_response(self, path: Path):
        if not self._needs_browser_conversion(path):
            return FileResponse(path)
        if path.suffix.lower() in (".heic", ".heif") and pillow_heif is None:
            raise HTTPException(status_code=415, detail="HEIC/HEIF 需要安装 pillow-heif")
        try:
            with Image.open(path) as image:
                if image.mode in ("RGBA", "LA") or (
                    image.mode == "P" and "transparency" in image.info
                ):
                    canvas = Image.new("RGB", image.size, (255, 255, 255))
                    canvas.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
                    image = canvas
                else:
                    image = image.convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=92, optimize=True)
        except Exception as error:
            raise HTTPException(status_code=415, detail=f"图片转换失败: {error}") from error
        return Response(buffer.getvalue(), media_type="image/jpeg")

    def _image_files(self):
        return sorted(
            (path for path in self.workspace.rglob("*") if self._is_image(path)),
            key=lambda path: self._relative(path).lower(),
        )

    def _directory_tree(self, path: Path, include_children: bool = True):
        entries = list(path.iterdir())
        directories = sorted((item for item in entries if item.is_dir()), key=lambda item: item.name.lower())
        children = [self._directory_tree(child, include_children=False) for child in directories] if include_children else []
        images = [item for item in entries if self._is_image(item)]
        direct_complete = all(
            not self._is_damaged_image(image)
            and self._validate_label_file(image.with_suffix(".txt")) == "valid"
            for image in images
        )
        has_images = bool(images)
        complete = has_images and direct_complete
        return {
            "name": path.name if path != self.workspace else self.workspace.name,
            "path": "" if path == self.workspace else self._relative(path),
            "has_images": has_images,
            "complete": complete,
            "has_child_dirs": bool(directories),
            "children_loaded": include_children,
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

    def _label_statistics(self, label_file: Path):
        result = {"status": "missing", "classes": {}}
        if not label_file.exists():
            return result
        try:
            lines = [
                line.strip()
                for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except OSError:
            result["status"] = "invalid"
            return result
        if not lines:
            result["status"] = "empty"
            return result
        valid = True
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                valid = False
                continue
            try:
                class_id = int(parts[0])
                [float(value) for value in parts[1:]]
            except ValueError:
                valid = False
                continue
            label = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
            result["classes"][label] = result["classes"].get(label, 0) + 1
        result["status"] = "valid" if valid else "invalid"
        return result

    def _project_index_file(self):
        if not self.log_root:
            return None
        return Path(self.log_root) / ".workstation" / "index.json"

    def _apply_index_label_change(self, old_stats: dict, new_stats: dict, image_delta: int = 0):
        index_file = self._project_index_file()
        if index_file is None or not index_file.is_file():
            self._invalidate_statistics()
            return
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._invalidate_statistics()
            return
        annotate = index.setdefault("annotate", {})
        annotate["images"] = max(0, int(annotate.get("images") or 0) + image_delta)
        status_keys = {
            "missing": "txt_missing",
            "empty": "txt_empty",
            "invalid": "txt_invalid",
            "valid": "txt_valid",
        }
        old_status = old_stats.get("status", "missing")
        new_status = new_stats.get("status", "missing")
        if old_status != "missing":
            annotate["labels"] = max(0, int(annotate.get("labels") or 0) - 1)
        if new_status != "missing":
            annotate["labels"] = int(annotate.get("labels") or 0) + 1
        for status, delta in ((old_status, -1), (new_status, 1)):
            key = status_keys.get(status)
            if key:
                annotate[key] = max(0, int(annotate.get(key) or 0) + delta)
        class_counts = annotate.setdefault("class_counts", {})
        for label, count in (old_stats.get("classes") or {}).items():
            next_count = int(class_counts.get(label) or 0) - int(count or 0)
            if next_count > 0:
                class_counts[label] = next_count
            else:
                class_counts.pop(label, None)
        for label, count in (new_stats.get("classes") or {}).items():
            class_counts[label] = int(class_counts.get(label) or 0) + int(count or 0)
        index["updated_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        self._invalidate_statistics()

    def _rebuild_workspace_index(self):
        index_file = self._project_index_file()
        if index_file is None or not index_file.is_file():
            self._invalidate_statistics()
            return
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {}
        annotate = {
            "images": 0,
            "labels": 0,
            "txt_missing": 0,
            "txt_empty": 0,
            "txt_invalid": 0,
            "txt_valid": 0,
            "class_counts": {},
        }
        for image_path in self.workspace.rglob("*"):
            if not self._is_image(image_path):
                continue
            annotate["images"] += 1
            stats = self._label_statistics(image_path.with_suffix(".txt"))
            status = stats["status"]
            if status != "missing":
                annotate["labels"] += 1
            key = {
                "missing": "txt_missing",
                "empty": "txt_empty",
                "invalid": "txt_invalid",
                "valid": "txt_valid",
            }.get(status)
            if key:
                annotate[key] += 1
            for label, count in stats["classes"].items():
                annotate["class_counts"][label] = annotate["class_counts"].get(label, 0) + count
        index["annotate"] = annotate
        index["updated_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        self._invalidate_statistics()

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
        indexed = self._indexed_statistics()
        if indexed is not None:
            return indexed
        if self.statistics_cache and time.time() - self.statistics_cache_at < 10:
            return self.statistics_cache
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
        self.statistics_cache = result
        self.statistics_cache_at = time.time()
        return result

    def _invalidate_statistics(self):
        self.statistics_cache = None
        self.statistics_cache_at = 0.0

    def _indexed_statistics(self):
        if not self.log_root:
            return None
        index_file = Path(self.log_root) / ".workstation" / "index.json"
        if not index_file.is_file():
            return None
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        annotate = index.get("annotate") or {}
        images = int(annotate.get("images") or 0)
        txt_missing = int(annotate.get("txt_missing") or 0)
        txt_empty = int(annotate.get("txt_empty") or 0)
        txt_invalid = int(annotate.get("txt_invalid") or 0)
        txt_valid = int(annotate.get("txt_valid") or 0)
        txt_total = txt_empty + txt_invalid + txt_valid
        return {
            "workspace": str(self.workspace),
            "images": images,
            "images_damaged": 0,
            "txt_total": txt_total,
            "txt_missing": txt_missing,
            "txt_empty": txt_empty,
            "txt_invalid": txt_invalid,
            "txt_valid": txt_valid,
            "txt_problem": txt_missing + txt_empty + txt_invalid,
            "txt_invalid_total": txt_empty + txt_invalid,
            "classes": len(self.classes),
            "classes_files": len(self.class_groups),
            "indexed": True,
            "updated_at": index.get("updated_at", ""),
        }

    def _models_root(self):
        return (self.model_root or self.workspace).resolve()

    def _model_files(self):
        root = self._models_root()
        models_dir = root / "models"
        if not models_dir.is_dir():
            return []
        exts = (".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel")
        models = []
        for path in sorted(models_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            stat = path.stat()
            models.append(
                {
                    "name": path.stem,
                    "filename": path.name,
                    "path": path.relative_to(root).as_posix(),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes"),
                }
            )
        return models

    def _safe_model_path(self, model_path: str):
        root = self._models_root()
        path = (root / (model_path or "")).resolve()
        if path == root or root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="model not found")
        models_dir = (root / "models").resolve()
        if path != models_dir and models_dir not in path.parents:
            raise HTTPException(status_code=400, detail="invalid model path")
        return path

    def _predict_boxes(self, image_path: str, model_path: str):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        model_file = self._safe_model_path(model_path)
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise HTTPException(status_code=500, detail="缺少 ultralytics，无法自动识别") from error
        try:
            model = self.model_cache.get(str(model_file))
            if model is None:
                model = YOLO(str(model_file))
                self.model_cache[str(model_file)] = model
            results = model.predict(str(path), verbose=False)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"自动识别失败: {error}") from error
        boxes = []
        max_classes = self._max_class_count()
        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None or result_boxes.cls is None or result_boxes.xywhn is None:
                continue
            classes = result_boxes.cls.cpu().tolist()
            coords = result_boxes.xywhn.cpu().tolist()
            for index, xywh in enumerate(coords):
                class_id = int(classes[index])
                if max_classes and class_id >= max_classes:
                    continue
                label = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
                boxes.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "cx": float(xywh[0]),
                        "cy": float(xywh[1]),
                        "width": float(xywh[2]),
                        "height": float(xywh[3]),
                    }
                )
        return {
            "image": self._relative(path),
            "model": model_file.relative_to(self._models_root()).as_posix(),
            "boxes": boxes,
        }

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
        old_stats = self._label_statistics(label_file)
        if lines:
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            label_file.write_text("", encoding="utf-8")
        self._apply_index_label_change(old_stats, self._label_statistics(label_file))
        if self.team_mode:
            self._append_operation_log(username or "未命名", relative_path, lines)
        if self.team_mode and client_id:
            self._release_lock(relative_path, client_id)
        return self._read_annotation(image_path)

    def _delete_image_file(self, image_path: str, client_id: str = "", username: str = ""):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        relative_path = self._relative(path)
        if self.team_mode:
            lock = self._lock_for(relative_path)
            if lock and lock["client_id"] != client_id:
                raise HTTPException(status_code=423, detail=f"文件已被 {lock['username']} 锁定")
        label_file = path.with_suffix(".txt")
        old_stats = self._label_statistics(label_file)
        deleted = []
        for item in (path, label_file):
            if item.exists():
                item.unlink()
                deleted.append(self._relative(item))
        self._apply_index_label_change(old_stats, {"status": "missing", "classes": {}}, image_delta=-1)
        if self.team_mode:
            timestamp = datetime.now().isoformat(timespec="seconds")
            message = f"[{timestamp}] {username or '未命名'} 删除 {relative_path}: {', '.join(deleted)}\n"
            try:
                with open(self._log_file(), "a", encoding="utf-8") as file:
                    file.write(message)
            except OSError:
                pass
        if self.team_mode and client_id:
            self._release_lock(relative_path, client_id)
        return {"deleted": deleted}

    def _directory_operation(self, directory: str, action: str, name: str = ""):
        path = self._safe_path(directory)
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="directory not found")
        if action == "rename":
            if path == self.workspace:
                raise HTTPException(status_code=400, detail="不能重命名根目录")
            renamed = self._rename_path(path, name)
            self._rebuild_workspace_index()
            return {"ok": True, "action": action, "renamed": renamed}
        if action == "delete_directory":
            if path == self.workspace:
                raise HTTPException(status_code=400, detail="不能删除根目录")
            shutil.rmtree(path)
            self._rebuild_workspace_index()
            return {"ok": True, "action": action, "deleted": self._relative(path)}
        if action == "delete_txt":
            deleted = []
            for item in sorted(path.rglob("*.txt"), key=lambda item: item.as_posix().lower()):
                if item.is_file() and item.name.lower() != "classes.txt":
                    item.unlink()
                    deleted.append(self._relative(item))
            self._rebuild_workspace_index()
            return {"ok": True, "action": action, "deleted": deleted}
        if action == "create_negative_txt":
            created = []
            for image in sorted((item for item in path.rglob("*") if self._is_image(item)), key=lambda item: item.as_posix().lower()):
                label_file = image.with_suffix(".txt")
                if not label_file.exists():
                    label_file.write_text("", encoding="utf-8")
                    created.append(self._relative(label_file))
            self._rebuild_workspace_index()
            return {"ok": True, "action": action, "created": created}
        if action in ("lowercase", "uppercase"):
            changed = []
            skipped = []
            items = sorted(
                (item for item in path.rglob("*") if item.is_file()),
                key=lambda item: len(item.parts),
                reverse=True,
            )
            for item in items:
                next_name = item.name.lower() if action == "lowercase" else item.name.upper()
                if next_name == item.name:
                    continue
                target = item.with_name(next_name)
                if target.exists():
                    try:
                        same_file = item.samefile(target)
                    except OSError:
                        same_file = False
                    if not same_file:
                        skipped.append(self._relative(item))
                        continue
                    temporary = item.with_name(f".{item.name}.yoloutils-rename-tmp")
                    suffix = 0
                    while temporary.exists():
                        suffix += 1
                        temporary = item.with_name(f".{item.name}.yoloutils-rename-tmp-{suffix}")
                    item.rename(temporary)
                    temporary.rename(target)
                    changed.append({"from": self._relative(item), "to": self._relative(target)})
                    continue
                item.rename(target)
                changed.append({"from": self._relative(item), "to": self._relative(target)})
            self._rebuild_workspace_index()
            return {"ok": True, "action": action, "changed": changed, "skipped": skipped}
        raise HTTPException(status_code=400, detail="invalid action")

    def _rename_path(self, path: Path, name: str):
        name = (name or "").strip()
        if not name or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="invalid name")
        target = path.with_name(name)
        if target.exists():
            raise HTTPException(status_code=409, detail="目标名称已存在")
        path.rename(target)
        return {"from": self._relative(path), "to": self._relative(target)}

    def _file_operation(self, image_path: str, action: str, name: str = ""):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        label_file = path.with_suffix(".txt")
        if action == "delete_txt":
            old_stats = self._label_statistics(label_file)
            deleted = []
            if label_file.exists():
                label_file.unlink()
                deleted.append(self._relative(label_file))
            self._apply_index_label_change(old_stats, {"status": "missing", "classes": {}})
            return {"ok": True, "action": action, "deleted": deleted}
        if action == "create_negative_txt":
            old_stats = self._label_statistics(label_file)
            created = []
            if not label_file.exists():
                label_file.write_text("", encoding="utf-8")
                created.append(self._relative(label_file))
            self._apply_index_label_change(old_stats, self._label_statistics(label_file))
            return {"ok": True, "action": action, "created": created}
        if action in ("lowercase", "uppercase"):
            changed = []
            for item in (path, label_file):
                if not item.exists():
                    continue
                next_name = item.name.lower() if action == "lowercase" else item.name.upper()
                if next_name == item.name:
                    continue
                changed.extend(self._rename_case(item, next_name))
            return {"ok": True, "action": action, "changed": changed}
        if action == "rename":
            old_label = label_file
            target_name = (name or "").strip()
            if not target_name or "/" in target_name or "\\" in target_name:
                raise HTTPException(status_code=400, detail="invalid name")
            target_image = path.with_name(target_name)
            if old_label.exists() and target_image.with_suffix(".txt").exists():
                raise HTTPException(status_code=409, detail="目标 .txt 已存在")
            renamed = self._rename_path(path, target_name)
            new_image = self._safe_path(renamed["to"])
            if old_label.exists():
                new_label = new_image.with_suffix(".txt")
                old_label.rename(new_label)
                renamed["label"] = {"from": self._relative(old_label), "to": self._relative(new_label)}
            return {"ok": True, "action": action, "renamed": renamed}
        raise HTTPException(status_code=400, detail="invalid action")

    def _rename_case(self, item: Path, next_name: str):
        target = item.with_name(next_name)
        if target.exists():
            try:
                same_file = item.samefile(target)
            except OSError:
                same_file = False
            if not same_file:
                raise HTTPException(status_code=409, detail=f"目标名称已存在: {target.name}")
            temporary = item.with_name(f".{item.name}.yoloutils-rename-tmp")
            suffix = 0
            while temporary.exists():
                suffix += 1
                temporary = item.with_name(f".{item.name}.yoloutils-rename-tmp-{suffix}")
            item.rename(temporary)
            temporary.rename(target)
            return [{"from": self._relative(item), "to": self._relative(target)}]
        item.rename(target)
        return [{"from": self._relative(item), "to": self._relative(target)}]

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
                    "使用 -d/--daemon 后，日志会写入 .project.log。",
                ],
            }
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            return {"file": str(log_file), "lines": [f"读取日志失败: {error}"]}
        return {"file": str(log_file), "lines": content[-lines:]}

    def _create_app(self):
        app = FastAPI(title="Yolo Workstation")
        static_dir = Path(__file__).resolve().parent / "static"
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(self._html())

        @app.get("/api/config")
        def config():
            return {"team_mode": self.team_mode, "share_url": self._share_url() if self.team_mode else ""}

        @app.get("/api/tree")
        def tree(directory: str = Query(default="")):
            path = self._safe_path(directory)
            if not path.is_dir():
                raise HTTPException(status_code=404, detail="directory not found")
            return self._directory_tree(path)

        @app.get("/api/files")
        def files(directory: str = Query(default="")):
            return {"directory": directory, "files": self._list_files(directory)}

        @app.get("/api/models")
        def models():
            return {"models": self._model_files()}

        @app.get("/api/classes")
        def classes():
            return {
                "classes_file": self.class_groups[0]["classes_file"] if self.class_groups else None,
                "classes": self.classes,
                "content": self._classes_text(),
                "class_groups": [
                    {
                        "classes_file": group["classes_file"],
                        "classes": group["classes"],
                    }
                    for group in self.class_groups
                ],
            }

        @app.post("/api/classes")
        async def save_classes(request: Request):
            payload = await request.json()
            target = self._save_classes_text(str(payload.get("content", "")))
            return {
                "ok": True,
                "classes_file": self._relative(target) if self._is_inside_workspace(target) else str(target),
                "classes": self.classes,
                "content": self._classes_text(),
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
            self._online_count()
            username_key = username.casefold()
            for current_client_id, info in self.presence.items():
                if current_client_id == client_id:
                    continue
                if str(info.get("username", "")).casefold() == username_key:
                    raise HTTPException(status_code=409, detail="用户名已被使用")
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

        @app.post("/api/auto-annotate")
        async def auto_annotate(request: Request):
            payload = await request.json()
            return self._predict_boxes(
                payload.get("path", ""),
                str(payload.get("model", "")).strip(),
            )

        @app.post("/api/file/delete")
        async def delete_file(request: Request):
            payload = await request.json()
            return self._delete_image_file(
                payload.get("path", ""),
                str(payload.get("client_id", "")).strip(),
                str(payload.get("username", "")).strip(),
            )

        @app.post("/api/directory/action")
        async def directory_action(request: Request):
            payload = await request.json()
            return self._directory_operation(
                str(payload.get("path", "")).strip(),
                str(payload.get("action", "")).strip(),
                str(payload.get("name", "")).strip(),
            )

        @app.post("/api/file/action")
        async def file_action(request: Request):
            payload = await request.json()
            return self._file_operation(
                str(payload.get("path", "")).strip(),
                str(payload.get("action", "")).strip(),
                str(payload.get("name", "")).strip(),
            )

        @app.get("/api/exif")
        def exif(path: str):
            return self._read_exif(path)

        @app.get("/media")
        def media(path: str, raw: bool = Query(default=False)):
            file_path = self._safe_path(path)
            if not self._is_image(file_path):
                raise HTTPException(status_code=404, detail="image not found")
            if raw:
                return FileResponse(file_path)
            return self._image_response(file_path)

        return app

    def _html(self):
        template_root = Path(__file__).resolve().parent / "templates"
        framework = (template_root / "framework.html").read_text(encoding="utf-8")
        annotate = (template_root / "annotate" / "index.html").read_text(encoding="utf-8")
        html = framework.replace("__ANNOTATE_CONTENT__", annotate)
        return html.replace(
            "__BODY_CLASS__",
            "team-mode username-required" if self.team_mode else "",
        ).replace("__TEAM_MODE__", "true" if self.team_mode else "false")

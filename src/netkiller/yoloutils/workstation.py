import os
import subprocess
import sys
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
    def __init__(self):
        self.workspace = None
        self.classes_file = None
        self.classes = []

    def main(
        self,
        workspace: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        daemon: bool = False,
    ):
        if FastAPI is None or uvicorn is None:
            print("缺少依赖: fastapi/uvicorn，请先安装: pip install fastapi uvicorn")
            return

        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            print(f"workspace 目录不存在: {self.workspace}")
            return

        if daemon:
            self._start_daemon(host, port)
            return

        self.classes_file = self._find_classes_file()
        self.classes = self._load_classes()
        app = self._create_app()

        print(f"Yolo Workstation: http://{host}:{port}")
        print(f"Workspace: {self.workspace}")
        if self.classes_file:
            print(f"Classes: {self.classes_file}")
        else:
            print("Classes: 未找到 classes.txt")
        uvicorn.run(app, host=host, port=port)

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

    def _start_daemon(self, host: str, port: int):
        pid = self._existing_pid()
        if pid is not None:
            print(f"Yolo Workstation 已在后台运行: pid={pid}")
            print(f"Yolo Workstation: http://{host}:{port}")
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
        print(f"Yolo Workstation: http://{host}:{port}")
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

    def _find_classes_file(self):
        files = sorted(self.workspace.rglob("classes.txt"))
        return files[0] if files else None

    def _load_classes(self):
        if not self.classes_file:
            return []
        with open(self.classes_file, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

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
        return {
            "name": path.name if path != self.workspace else self.workspace.name,
            "path": "" if path == self.workspace else self._relative(path),
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
            files.append(
                {
                    "name": item.name,
                    "path": self._relative(item),
                    "label": self._relative(label_file) if label_file.exists() else None,
                    "label_status": label_status,
                    "damaged": damaged,
                }
            )
        return files

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
            if class_id < 0 or class_id >= len(self.classes):
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

    def _write_annotation(self, image_path: str, boxes):
        path = self._safe_path(image_path)
        if not self._is_image(path):
            raise HTTPException(status_code=404, detail="image not found")
        if not isinstance(boxes, list):
            raise HTTPException(status_code=400, detail="boxes must be a list")

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
            if class_id < 0 or (self.classes and class_id >= len(self.classes)):
                raise HTTPException(status_code=400, detail="invalid class_id")
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

        label_file = path.with_suffix(".txt")
        if lines:
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            label_file.write_text("", encoding="utf-8")
        return self._read_annotation(image_path)

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

    def _create_app(self):
        app = FastAPI(title="Yolo Workstation")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(self._html())

        @app.get("/api/tree")
        def tree():
            return self._directory_tree(self.workspace)

        @app.get("/api/files")
        def files(directory: str = Query(default="")):
            return {"directory": directory, "files": self._list_files(directory)}

        @app.get("/api/classes")
        def classes():
            return {
                "classes_file": self._relative(self.classes_file) if self.classes_file else None,
                "classes": self.classes,
            }

        @app.get("/api/statistics")
        def statistics():
            return self._statistics()

        @app.get("/api/annotation")
        def annotation(path: str):
            return self._read_annotation(path)

        @app.post("/api/annotation")
        async def save_annotation(request: Request):
            payload = await request.json()
            return self._write_annotation(payload.get("path", ""), payload.get("boxes", []))

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
    header { height: 48px; display: flex; align-items: center; padding: 0 16px; border-bottom: 1px solid #d9e2ec; background: #fff; font-weight: 650; }
    main { height: calc(100vh - 88px); display: grid; grid-template-columns: minmax(360px, 520px) 8px minmax(420px, 1fr) 220px; overflow: hidden; }
    main.tree-hidden { grid-template-columns: 280px 8px minmax(420px, 1fr) 220px; }
    main.right-hidden { grid-template-columns: minmax(360px, 520px) 8px minmax(420px, 1fr) 0; }
    main.tree-hidden.right-hidden { grid-template-columns: 280px 8px minmax(420px, 1fr) 0; }
    aside, section { min-width: 0; overflow: auto; border-right: 1px solid #d9e2ec; background: #fff; }
    section.viewer { background: #f5f7fa; display: flex; flex-direction: column; }
    h2 { margin: 0; padding: 12px 14px; font-size: 13px; border-bottom: 1px solid #e4e7eb; background: #f8fafc; }
    .pane-header { flex: 0 0 42px; height: 42px; min-height: 42px; max-height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 0 10px 0 14px; overflow: hidden; border-bottom: 1px solid #e4e7eb; background: #f8fafc; }
    .pane-header h2 { flex: 1 1 auto; min-width: 0; padding: 0; overflow: hidden; border: 0; background: transparent; line-height: 1; white-space: nowrap; text-overflow: ellipsis; }
    .icon-button { flex: 0 0 28px; width: 28px; min-width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; padding: 0; text-align: center; border-radius: 6px; font-size: 16px; color: #52606d; }
    .icon-button:hover { background: #e6f0ff; color: #243b53; }
    .viewer-tools { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
    .tool-button { width: auto; min-width: 34px; height: 28px; padding: 0 8px; display: inline-flex; align-items: center; justify-content: center; border: 1px solid #d9e2ec; border-radius: 6px; background: #fff; color: #334e68; font-size: 12px; line-height: 1; white-space: nowrap; }
    .tool-button:hover { background: #e6f0ff; color: #243b53; }
    .tool-button.active { border-color: #2563eb; background: #dbeafe; color: #1d4ed8; }
    .left-panel { display: grid; grid-template-columns: minmax(120px, 46%) 8px minmax(160px, 1fr); overflow: hidden; border-right: 1px solid #d9e2ec; background: #fff; }
    .left-panel.tree-hidden { display: block; }
    .left-panel.tree-hidden .files-pane { height: 100%; }
    .left-pane { min-width: 0; overflow: hidden; background: #fff; display: flex; flex-direction: column; }
    .left-panel.tree-hidden .tree-pane, .left-panel.tree-hidden .vertical-splitter { display: none; }
    .vertical-splitter { cursor: col-resize; background: #e4e7eb; border-left: 1px solid #d9e2ec; border-right: 1px solid #d9e2ec; }
    .vertical-splitter:hover, .vertical-splitter.dragging { background: #bcccdc; }
    .main-splitter { cursor: col-resize; background: #e4e7eb; border-left: 1px solid #d9e2ec; border-right: 1px solid #d9e2ec; }
    .main-splitter:hover, .main-splitter.dragging { background: #bcccdc; }
    .tree, .files, .labels, .exif { padding: 8px; }
    .tree, .files { flex: 1 1 auto; min-height: 0; overflow: auto; }
    button { width: 100%; border: 0; background: transparent; text-align: left; padding: 7px 8px; border-radius: 6px; cursor: pointer; color: #243b53; font: inherit; }
    button:hover, button.active { background: #e6f0ff; }
    .tree button { padding-left: calc(8px + var(--level) * 14px); }
    .file-meta { display: block; margin-top: 2px; color: #7b8794; font-size: 12px; }
    .file-valid { color: #166534; }
    .file-valid .file-meta { color: #15803d; }
    .file-invalid { color: #991b1b; }
    .file-invalid .file-meta { color: #dc2626; }
    .file-valid.active { background: #dcfce7; }
    .file-invalid.active { background: #fee2e2; }
    .canvas-wrap { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; padding: 16px; overflow: auto; }
    #canvas { max-width: 100%; max-height: 100%; background: #fff; box-shadow: 0 1px 8px rgba(31, 41, 51, .18); }
    .empty { color: #7b8794; padding: 16px; }
    .label-row { display: flex; align-items: center; gap: 8px; padding: 6px 4px; font-size: 13px; }
    .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    .right-panel { display: flex; flex-direction: column; overflow: hidden; }
    main.right-hidden .right-panel { display: none; }
    .right-pane { min-height: 80px; overflow: auto; }
    .top-info-pane { flex: 0 0 40%; min-height: 180px; display: flex; flex-direction: column; overflow: hidden; }
    .labels-pane { flex: 1 1 55%; min-height: 42px; overflow: hidden; display: flex; flex-direction: column; }
    .labels { flex: 1 1 auto; min-height: 0; overflow: auto; }
    .histogram-pane { flex: 1 1 45%; min-height: 42px; overflow: hidden; display: flex; flex-direction: column; }
    .right-panel.histogram-collapsed .labels-pane { flex-grow: 0; }
    .right-panel.histogram-collapsed .histogram-pane { flex: 0 0 42px; }
    .right-panel.histogram-collapsed .histogram-splitter { display: none; }
    .right-panel.histogram-collapsed .histogram { display: none; }
    .right-pane.exif-pane { flex: 1 1 auto; }
    .right-panel.exif-collapsed .top-info-pane { flex: 1 1 auto; }
    .right-panel.exif-collapsed .splitter { display: none; }
    .right-panel.exif-collapsed .exif-pane { flex: 0 0 42px; min-height: 42px; overflow: hidden; }
    .right-panel.exif-collapsed .exif { display: none; }
    .histogram { flex: 1 1 auto; min-height: 0; padding: 8px; }
    #histogramCanvas { width: 100%; height: 100%; display: block; background: #fff; border: 1px solid #e4e7eb; border-radius: 4px; }
    .splitter { flex: 0 0 8px; cursor: row-resize; background: #e4e7eb; border-top: 1px solid #d9e2ec; border-bottom: 1px solid #d9e2ec; }
    .splitter:hover, .splitter.dragging { background: #bcccdc; }
    .histogram-splitter { flex: 0 0 8px; cursor: row-resize; background: #edf2f7; border-top: 1px solid #d9e2ec; border-bottom: 1px solid #d9e2ec; }
    .histogram-splitter:hover, .histogram-splitter.dragging { background: #bcccdc; }
    .kv-row { display: grid; grid-template-columns: minmax(76px, 38%) minmax(0, 1fr); gap: 8px; padding: 6px 4px; border-bottom: 1px solid #edf2f7; font-size: 12px; }
    .kv-key { color: #52606d; overflow-wrap: anywhere; }
    .kv-value { color: #1f2933; overflow-wrap: anywhere; }
    footer { height: 40px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 16px; border-top: 1px solid #d9e2ec; background: #fff; font-size: 13px; color: #52606d; }
    .stats-left { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .stats-right { flex: 0 0 auto; display: flex; align-items: center; justify-content: flex-end; gap: 18px; white-space: nowrap; }
    .stat strong { color: #1f2933; font-weight: 700; }
    .stat.warn strong { color: #b45309; }
    .stat.bad strong { color: #be123c; }
  </style>
</head>
<body>
  <header>Yolo Workstation</header>
  <main id="appMain">
    <div id="leftPanel" class="left-panel">
      <aside id="treePane" class="left-pane tree-pane">
        <div class="pane-header"><h2>目录</h2><button id="toggleTree" class="icon-button" title="隐藏目录栏">‹</button></div>
        <div id="tree" class="tree"></div>
      </aside>
      <div id="leftSplitter" class="vertical-splitter" title="拖动调整目录和文件列表比例"></div>
      <aside class="left-pane files-pane">
        <div class="pane-header"><h2>文件</h2><button id="showTree" class="icon-button" title="显示目录栏">☰</button></div>
        <div id="files" class="files"></div>
      </aside>
    </div>
    <div id="mainSplitter" class="main-splitter" title="拖动调整文件列表和图像区域比例"></div>
    <section class="viewer">
      <div class="pane-header">
        <h2 id="viewerTitle">图像</h2>
        <div class="viewer-tools">
          <button id="autoAnnotate" class="tool-button" title="自动标注激活">自动</button>
          <button id="deleteAnnotation" class="tool-button" title="删除当前标注">删除</button>
          <button id="resetAnnotation" class="tool-button" title="重新读取标注">重置</button>
          <button id="saveAnnotation" class="tool-button" title="保存当前标注">保存</button>
          <button id="showRight" class="icon-button" title="显示标签/EXIF 栏">☰</button>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="canvas"></canvas><div id="empty" class="empty">请选择图片</div></div>
    </section>
    <aside id="rightPanel" class="right-panel">
      <div id="topInfoPane" class="right-pane top-info-pane">
        <div id="labelsPane" class="labels-pane">
          <div class="pane-header"><h2>标签</h2><button id="hideRight" class="icon-button" title="隐藏标签/EXIF 栏">›</button></div>
          <div id="labels" class="labels"></div>
        </div>
        <div id="histogramSplitter" class="histogram-splitter" title="拖动调整标签和直方图比例"></div>
        <div id="histogramPane" class="histogram-pane">
          <div class="pane-header"><h2>直方图</h2><button id="toggleHistogram" class="icon-button" title="折叠/展开直方图">⌄</button></div>
          <div id="histogram" class="histogram"><canvas id="histogramCanvas"></canvas></div>
        </div>
      </div>
      <div id="splitter" class="splitter" title="拖动调整标签和 EXIF 面板比例"></div>
      <div id="exifPane" class="right-pane exif-pane">
        <div class="pane-header"><h2>EXIF</h2><button id="toggleExif" class="icon-button" title="折叠/展开 EXIF">⌄</button></div>
        <div id="exif" class="exif"><div class="empty">请选择图片</div></div>
      </div>
    </aside>
  </main>
  <footer id="stats">
    <div class="stats-left">位置：-</div>
    <div class="stats-right">
      <span class="stat">图像 <strong>-</strong></span>
      <span class="stat">.txt <strong>-</strong></span>
      <span class="stat bad">损坏图像 <strong>-</strong></span>
      <span class="stat bad">无效 .txt <strong>-</strong></span>
    </div>
  </footer>
  <script>
    const colors = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f"];
    const treeEl = document.getElementById("tree");
    const filesEl = document.getElementById("files");
    const appMain = document.getElementById("appMain");
    const leftPanel = document.getElementById("leftPanel");
    const leftSplitter = document.getElementById("leftSplitter");
    const mainSplitter = document.getElementById("mainSplitter");
    const toggleTree = document.getElementById("toggleTree");
    const showTree = document.getElementById("showTree");
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
    const deleteAnnotation = document.getElementById("deleteAnnotation");
    const resetAnnotation = document.getElementById("resetAnnotation");
    const saveAnnotation = document.getElementById("saveAnnotation");
    const toggleHistogram = document.getElementById("toggleHistogram");
    const toggleExif = document.getElementById("toggleExif");
    const statsEl = document.getElementById("stats");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const empty = document.getElementById("empty");
    const viewerTitle = document.getElementById("viewerTitle");
    let currentDir = "";
    let currentPath = "";
    let currentImage = null;
    let currentBoxes = [];
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

    function renderTree(node, level = 0) {
      const button = document.createElement("button");
      button.textContent = node.name || "/";
      button.style.setProperty("--level", level);
      button.dataset.path = node.path;
      button.onclick = () => selectDir(node.path, button);
      treeEl.appendChild(button);
      for (const child of node.children) renderTree(child, level + 1);
    }

    async function selectDir(path, button) {
      currentDir = path;
      document.querySelectorAll("#tree button").forEach(item => item.classList.remove("active"));
      if (button) button.classList.add("active");
      const data = await getJson(`/api/files?directory=${encodeURIComponent(path)}`);
      filesEl.innerHTML = "";
      if (!data.files.length) {
        filesEl.innerHTML = '<div class="empty">没有图片</div>';
        return;
      }
      for (const file of data.files) {
        const row = document.createElement("button");
        const isInvalid = file.damaged || file.label_status === "empty" || file.label_status === "invalid";
        const isValid = !file.damaged && file.label_status === "valid";
        if (isInvalid) row.classList.add("file-invalid");
        if (isValid) row.classList.add("file-valid");
        const statusText = file.damaged
          ? "损坏图像"
          : file.label_status === "valid"
            ? "已标注"
            : file.label_status === "missing"
              ? "未标注"
              : file.label_status === "empty"
                ? "空 .txt"
                : "无效 .txt";
        row.innerHTML = `${escapeHtml(file.name)}<span class="file-meta">${statusText}</span>`;
        row.onclick = () => selectImage(file.path, row);
        filesEl.appendChild(row);
      }
    }

    async function selectImage(path, button) {
      document.querySelectorAll("#files button").forEach(item => item.classList.remove("active"));
      if (button) button.classList.add("active");
      viewerTitle.textContent = path;
      currentPath = path;
      const annotation = await getJson(`/api/annotation?path=${encodeURIComponent(path)}`);
      currentBoxes = annotation.boxes;
      loadExif(path);
      const image = new Image();
      image.onload = () => {
        currentImage = image;
        empty.style.display = "none";
        canvas.style.display = "block";
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        drawHistogram(image);
        redrawImage();
      };
      image.src = `/media?path=${encodeURIComponent(path)}`;
    }

    function redrawImage() {
      if (!currentImage) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(currentImage, 0, 0);
      drawBoxes(currentBoxes, currentImage.naturalWidth, currentImage.naturalHeight);
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

    function drawBoxes(boxes, imageWidth, imageHeight) {
      ctx.lineWidth = Math.max(2, Math.round(Math.min(imageWidth, imageHeight) / 360));
      ctx.font = `${Math.max(14, Math.round(imageWidth / 80))}px sans-serif`;
      for (const box of boxes) {
        const color = colors[Math.abs(box.class_id) % colors.length];
        const x = (box.cx - box.width / 2) * imageWidth;
        const y = (box.cy - box.height / 2) * imageHeight;
        const width = box.width * imageWidth;
        const height = box.height * imageHeight;
        ctx.strokeStyle = color;
        ctx.strokeRect(x, y, width, height);
        const text = box.label;
        const metrics = ctx.measureText(text);
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - 22), metrics.width + 10, 22);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x + 5, Math.max(16, y - 6));
      }
    }

    async function loadLabels() {
      const data = await getJson("/api/classes");
      labelsEl.innerHTML = "";
      if (data.classes_file) {
        const source = document.createElement("div");
        source.className = "file-meta";
        source.textContent = data.classes_file;
        labelsEl.appendChild(source);
      }
      data.classes.forEach((label, index) => {
        const row = document.createElement("div");
        row.className = "label-row";
        row.innerHTML = `<span class="swatch" style="background:${colors[index % colors.length]}"></span><span>${index}: ${label}</span>`;
        labelsEl.appendChild(row);
      });
      if (!data.classes.length) labelsEl.innerHTML = '<div class="empty">未找到 classes.txt</div>';
    }

    async function loadStatistics() {
      const data = await getJson("/api/statistics");
      const workspace = escapeHtml(data.workspace);
      statsEl.innerHTML = `
        <div class="stats-left" title="${workspace}">位置：${workspace}</div>
        <div class="stats-right">
          <span class="stat">图像 <strong>${data.images}</strong></span>
          <span class="stat">.txt <strong>${data.txt_total}</strong></span>
          <span class="stat bad">损坏图像 <strong>${data.images_damaged}</strong></span>
          <span class="stat bad">无效 .txt <strong>${data.txt_invalid_total}</strong></span>
        </div>
      `;
    }

    async function init() {
      canvas.style.display = "none";
      const tree = await getJson("/api/tree");
      renderTree(tree);
      await loadLabels();
      await loadStatistics();
      initSplitter();
      initHistogramSplitter();
      initLeftPanel();
      initMainSplitter();
      initRightPanel();
      initViewerTools();
      const rootButton = document.querySelector("#tree button");
      if (rootButton) await selectDir("", rootButton);
    }

    function initViewerTools() {
      autoAnnotate.addEventListener("click", () => {
        autoAnnotate.classList.toggle("active");
      });
      deleteAnnotation.addEventListener("click", () => {
        if (!currentPath) return;
        currentBoxes = [];
        redrawImage();
      });
      resetAnnotation.addEventListener("click", async () => {
        if (!currentPath) return;
        const annotation = await getJson(`/api/annotation?path=${encodeURIComponent(currentPath)}`);
        currentBoxes = annotation.boxes;
        redrawImage();
      });
      saveAnnotation.addEventListener("click", async () => {
        if (!currentPath) return;
        const annotation = await postJson("/api/annotation", {
          path: currentPath,
          boxes: currentBoxes,
        });
        currentBoxes = annotation.boxes;
        redrawImage();
        await loadStatistics();
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
</html>"""

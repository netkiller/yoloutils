#!/usr/bin/env python3
import argparse
import base64
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Yolo Workstation FastAPI 启动程序",
        epilog="Yolo workstation",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("-p", "--port", type=int, default=8000, help="监听端口")
    parser.add_argument("-d", "--daemon", action="store_true", default=False, help="后台运行")
    parser.add_argument("-w", "--workspace", type=str, default=None, help="标注工作目录")
    parser.add_argument("-s", "--datasets", type=str, default=None, help="数据集目录")
    parser.add_argument("-r", "--runs", type=str, default=None, help="训练目录")
    parser.add_argument("--open", action="store_true", default=False, help="启动服务后打开无地址栏应用窗口")
    parser.add_argument("-t", "--team", action="store_true", default=False, help="团队协作模式")
    parser.add_argument("--mDNS", dest="mdns", type=str, default=None, help=".local 分享域名")
    parser.add_argument("--reload", action="store_true", default=False, help="启用 uvicorn reload")
    parser.add_argument("--demo", action="store_true", default=False, help="演示模式")
    parser.add_argument(
        "--edition",
        choices=("community", "enterprise"),
        default="community",
        help="版本类型: community=社区版, enterprise=企业版",
    )
    parser.add_argument("--auth", dest="auth", type=str, default=None, help="user:password")
    return parser


def server_url(host: str, port: int):
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}"


def normalize_mdns(value: str):
    name = (value or "netkiller.local").strip().lower()
    if "://" in name:
        name = name.split("://", 1)[1]
    name = name.split("/", 1)[0].split(":", 1)[0]
    return name if name.endswith(".local") else f"{name}.local"


def apply_environment(args):
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path.cwd()
    if not workspace.is_dir():
        raise SystemExit(f"workspace 目录不存在: {workspace}")
    if args.auth:
        username, separator, password = args.auth.partition(":")
        if not separator or not username or not password:
            raise SystemExit("--auth 格式应为 user:password")

    os.environ["YOLOUTILS_WORKSPACE"] = str(workspace)
    os.environ["YOLOUTILS_HOST"] = str(args.host)
    os.environ["YOLOUTILS_PORT"] = str(args.port)
    os.environ["YOLOUTILS_EDITION"] = args.edition
    if args.mdns:
        os.environ["YOLOUTILS_MDNS"] = normalize_mdns(args.mdns)
    else:
        os.environ.pop("YOLOUTILS_MDNS", None)
    os.environ["YOLOUTILS_TEAM"] = "1" if args.team else "0"
    os.environ["YOLOUTILS_DEMO"] = "1" if args.demo else "0"
    if args.auth:
        os.environ["YOLOUTILS_AUTH"] = args.auth
    else:
        os.environ.pop("YOLOUTILS_AUTH", None)

    optional_paths = {
        "YOLOUTILS_DATASET": args.datasets,
        "YOLOUTILS_RUN": args.runs,
    }
    for key, value in optional_paths.items():
        if value:
            os.environ[key] = str(Path(value).expanduser().resolve())
        else:
            os.environ.pop(key, None)
    return workspace


def open_app_window(url: str):
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

    for command in (
            ["google-chrome", f"--app={url}", "--new-window"],
            ["chrome", f"--app={url}", "--new-window"],
            ["chromium", f"--app={url}", "--new-window"],
            ["chromium-browser", f"--app={url}", "--new-window"],
            ["microsoft-edge", f"--app={url}", "--new-window"],
            ["brave-browser", f"--app={url}", "--new-window"],
    ):
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            continue


def basic_auth_header(auth: str | None):
    if not auth:
        return {}
    token = base64.b64encode(auth.encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def start_browser_opener(url: str, auth: str | None = None):
    def open_and_keep_alive():
        time.sleep(1)
        open_app_window(url)
        while True:
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Yolo-Workstation-Headless/1.0",
                        **basic_auth_header(auth),
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read(2048)
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(
        target=open_and_keep_alive,
        daemon=True,
        name="yoloutils-browser-opener",
    ).start()


def daemon_command(args):
    command = [sys.executable, str(Path(__file__).resolve())]
    for option, attr in (
        ("host", "host"),
        ("port", "port"),
        ("workspace", "workspace"),
        ("datasets", "datasets"),
        ("runs", "runs"),
        ("mDNS", "mdns"),
        ("auth", "auth"),
        ("edition", "edition"),
    ):
        value = getattr(args, attr)
        if value is None:
            continue
        command.extend([f"--{option}", str(value)])
    if args.open:
        command.append("--open")
    if args.team:
        command.append("--team")
    if args.reload:
        command.append("--reload")
    if args.demo:
        command.append("--demo")
    return command


def start_daemon(args, workspace: Path):
    pid_file = workspace / ".yoloutils-workstation.pid"
    log_file = workspace / ".project.log"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            print(f"Yolo Workstation 已在后台运行: pid={pid}")
            print(f"Yolo Workstation: {server_url(args.host, args.port)}")
            print(f"PID: {pid_file}")
            print(f"LOG: {log_file}")
            return
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)

    with open(log_file, "ab") as output:
        process = subprocess.Popen(
            daemon_command(args),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            cwd=Path(__file__).resolve().parent,
            start_new_session=True,
            close_fds=True,
            env=os.environ.copy(),
        )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"Yolo Workstation 已后台启动: pid={process.pid}")
    print(f"Yolo Workstation: {server_url(args.host, args.port)}")
    print(f"PID: {pid_file}")
    print(f"LOG: {log_file}")


def check_dependencies():
    missing = []
    for package in ("fastapi", "jinja2", "uvicorn"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            f"缺少依赖: {', '.join(missing)}，请先执行: pip install -r requirements.txt"
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    workspace = apply_environment(args)
    url = server_url(args.host, args.port)

    if args.daemon:
        start_daemon(args, workspace)
        return

    check_dependencies()

    print(f"Yolo Workstation: {url}")
    print(f"Workspace: {workspace}")
    if args.datasets:
        print(f"Dataset: {Path(args.datasets).expanduser().resolve()}")
    if args.runs:
        print(f"Run: {Path(args.runs).expanduser().resolve()}")
    if args.demo:
        print("Demo: enabled")
    print(f"Edition: {args.edition}")
    if args.open:
        start_browser_opener(url, args.auth)
        print("Browser: opening")

    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

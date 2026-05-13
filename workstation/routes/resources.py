import hashlib
import io
import json
import posixpath
import stat as stat_module
import time
import asyncio
import subprocess
from urllib.parse import parse_qs, urlencode
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import compute_config, current_username, header_context, project_dir, require_team_login, team_mode_enabled, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def resources_file(workspace: Path):
    return workspace / ".resources.json"


def read_resources(workspace: Path):
    path = resources_file(workspace)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("resources", []) if isinstance(data, dict) else []
    return [normalize_resource(item) for item in items if isinstance(item, dict)]


def resource_key(item: dict):
    raw = f"{item.get('username', '')}@{item.get('host', '')}:{item.get('port', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def normalize_resource(item: dict):
    try:
        port = int(item.get("port") or 22)
    except (TypeError, ValueError):
        port = 22
    normalized = {
        "id": str(item.get("id") or resource_key(item)),
        "name": str(item.get("name") or item.get("host") or "未命名服务器"),
        "host": str(item.get("host") or ""),
        "port": port,
        "username": str(item.get("username") or ""),
        "password": str(item.get("password") or ""),
        "use_private_key": bool(item.get("use_private_key")),
        "private_key": str(item.get("private_key") or ""),
        "note": str(item.get("note") or ""),
        "summary": item.get("summary") if isinstance(item.get("summary"), dict) else default_resource_summary(),
    }
    normalized["address"] = f"{normalized['username']}@{normalized['host']}:{normalized['port']}"
    return normalized


def write_resources(workspace: Path, items: list[dict]):
    resources_file(workspace).write_text(
        json.dumps({"resources": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def current_project_from_request(request: Request, workspace: Path, project: str = ""):
    requested = project or request.cookies.get("current_project", "")
    return requested if project_dir(workspace, requested) else ""


def resources_base(current_project: str):
    return f"/resources/{current_project}" if current_project else "/resources"


def find_resource(workspace: Path, resource_id: str):
    for item in read_resources(workspace):
        if item["id"] == resource_id:
            return item
    return None


def default_resource_summary(error: str = ""):
    return {
        "ok": False,
        "error": error,
        "hostname": "未知",
        "system": "未知",
        "os_family": "linux",
        "cpu_count": "未知",
        "memory_total": "未知",
        "disk_total": "未知",
        "gpu_count": 0,
        "gpu_memory_total": "0 B",
    }


def resource_list_items(workspace: Path):
    return read_resources(workspace)


def resource_form_data(form: dict):
    name = (form.get("name", [""])[0] or "").strip()
    host = (form.get("host", [""])[0] or "").strip()
    username = (form.get("username", [""])[0] or "").strip()
    password = (form.get("password", [""])[0] or "").strip()
    use_private_key = (form.get("use_private_key", [""])[0] or "").lower() in ("1", "true", "yes", "on")
    private_key = (form.get("private_key", [""])[0] or "").strip()
    note = (form.get("note", [""])[0] or "").strip()
    try:
        port = int(form.get("port", ["22"])[0] or 22)
    except ValueError:
        port = 22
    port = max(1, min(port, 65535))
    return {
        "name": name or host,
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "use_private_key": use_private_key,
        "private_key": private_key if use_private_key else "",
        "note": note,
    }


def resource_detail_url(current_project: str, resource_id: str):
    return f"{resources_base(current_project)}/server/{resource_id}"


def resource_check_url(current_project: str, resource_id: str):
    return f"{resources_base(current_project)}/check?server={resource_id}"


def format_bytes(value: int | float):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return "0 B"


def capacity_metric(total: int, used: int, color: str = "#2563eb"):
    total = max(int(total or 0), 0)
    used = max(int(used or 0), 0)
    used = min(used, total) if total else 0
    available = max(total - used, 0)
    percent = round((used / total * 100) if total else 0, 1)
    return {
        "total": format_bytes(total),
        "used": format_bytes(used),
        "available": format_bytes(available),
        "percent": percent,
        "style": f"conic-gradient({color} 0 {percent}%, #e2e8f0 {percent}% 100%)",
    }


def segmented_capacity_metric(total: int, segments: list[tuple[str, int, str]]):
    total = max(int(total or 0), 0)
    cursor = 0.0
    gradient = []
    items = []
    used = 0
    for label, raw_value, color in segments:
        value = max(int(raw_value or 0), 0)
        if total:
            value = min(value, max(total - used, 0))
        percent = round((value / total * 100) if total else 0, 1)
        start = cursor
        end = min(cursor + percent, 100)
        if percent > 0:
            gradient.append(f"{color} {start}% {end}%")
        cursor = end
        used += value
        items.append({"label": label, "value": format_bytes(value), "bytes": value, "percent": percent, "color": color})
    if cursor < 100:
        gradient.append(f"#e2e8f0 {cursor}% 100%")
    return {
        "total": format_bytes(total),
        "used": format_bytes(used),
        "available": format_bytes(max(total - used, 0)),
        "percent": round((used / total * 100) if total else 0, 1),
        "items": items,
        "style": f"conic-gradient({', '.join(gradient)})" if gradient else "#e2e8f0",
    }


def run_ssh_commands(resource: dict):
    try:
        import paramiko
    except ImportError:
        return {
            "ok": False,
            "error": "当前 Python 环境未安装 paramiko，无法通过 SSH 获取远程指标。",
            "commands": {},
        }

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs = ssh_connect_kwargs(resource)
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **connect_kwargs,
        )
        commands = {
            "hostname": "hostname",
            "os_release": "awk -F= '/^PRETTY_NAME=/ {gsub(/\"/, \"\", $2); print $2}' /etc/os-release 2>/dev/null || true",
            "kernel": "uname -srmo",
            "cpu_count": "nproc 2>/dev/null || getconf _NPROCESSORS_ONLN",
            "cpu_counters": "grep '^cpu[0-9]' /proc/stat 2>/dev/null",
            "cpu_usage": "grep '^cpu[0-9]' /proc/stat 2>/dev/null | awk '{idle=$5+$6; total=0; for(i=2;i<=NF;i++) total+=$i; usage=(total-idle)*100/total; printf \"CPU %d %.1f\\n\", NR, usage}'",
            "loadavg": "cat /proc/loadavg 2>/dev/null || uptime",
            "uptime": "cat /proc/uptime 2>/dev/null | awk '{print int($1)}'",
            "memory": "free -b | awk '/Mem:/ {print $2\" \"$3\" \"$5\" \"$6\" \"$7}'",
            "disk": "df -B1 / | awk 'NR==2 {print $2\" \"$3\" \"$4\" \"$5}'",
            "disk_io": "awk '$3 !~ /^(loop|ram|sr)/ {r+=$6*512; w+=$10*512} END {print r+0\" \"w+0}' /proc/diskstats 2>/dev/null",
            "network": "awk -F'[: ]+' 'NR>2 && $2 != \"lo\" {rx+=$3; tx+=$11} END {print rx+0\" \"tx+0}' /proc/net/dev 2>/dev/null",
            "gpu": "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits || true",
        }
        output = {}
        for key, command in commands.items():
            stdin, stdout, stderr = client.exec_command(command, timeout=10)
            stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
            stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
            output[key] = stdout_text or stderr_text
        return {"ok": True, "error": "", "commands": output}
    except Exception as error:
        return {"ok": False, "error": f"SSH 连接失败：{error}", "commands": {}}
    finally:
        client.close()


def private_key_from_text(key_text: str, password: str = ""):
    import paramiko

    last_error = None
    key_classes = [
        key_class
        for key_class in (
            getattr(paramiko, "RSAKey", None),
            getattr(paramiko, "ECDSAKey", None),
            getattr(paramiko, "Ed25519Key", None),
            getattr(paramiko, "DSSKey", None),
        )
        if key_class is not None
    ]
    for key_class in key_classes:
        try:
            return key_class.from_private_key(io.StringIO(key_text), password=password or None)
        except Exception as error:
            last_error = error
    raise ValueError(f"私钥解析失败：{last_error}")


def ssh_connect_kwargs(resource: dict):
    if resource.get("use_private_key") and resource.get("private_key"):
        return {"pkey": private_key_from_text(resource["private_key"], resource.get("password", ""))}
    return {"password": resource.get("password", "")}


def remote_metrics(resource: dict):
    result = run_ssh_commands(resource)
    commands = result["commands"]
    cpu_count = 0
    try:
        cpu_count = int((commands.get("cpu_count") or "0").split()[0])
    except (ValueError, IndexError):
        cpu_count = 0

    load_values = []
    for value in (commands.get("loadavg") or "").split()[:3]:
        try:
            load_values.append(float(value))
        except ValueError:
            break
    while len(load_values) < 3:
        load_values.append(0.0)
    load_percent = round(min((load_values[0] / cpu_count * 100) if cpu_count else 0, 100), 1)
    load_items = [
        {
            "label": label,
            "value": value,
            "percent": round(min((value / cpu_count * 100) if cpu_count else 0, 100), 1),
        }
        for label, value in zip(("1 分钟", "5 分钟", "15 分钟"), load_values)
    ]

    cpu_items = []
    for index, line in enumerate((commands.get("cpu_usage") or "").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            percent = round(max(0, min(float(parts[-1]), 100)), 1)
        except ValueError:
            continue
        cpu_items.append({"label": f"CPU {index}", "percent": percent})
    if not cpu_items and cpu_count:
        cpu_items = [{"label": f"CPU {index}", "percent": 0.0} for index in range(1, cpu_count + 1)]

    cpu_counters = []
    for index, line in enumerate((commands.get("cpu_counters") or "").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            values = [int(value) for value in parts[1:]]
        except ValueError:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        cpu_counters.append({"label": f"CPU {index}", "idle": idle, "total": total})

    memory_parts = (commands.get("memory") or "").split()
    try:
        memory_total = int(memory_parts[0])
        memory_used = int(memory_parts[1])
        memory_shared = int(memory_parts[2])
        memory_cache = int(memory_parts[3])
        memory_available = int(memory_parts[4])
    except (ValueError, IndexError):
        memory_total, memory_used, memory_shared, memory_cache, memory_available = 0, 0, 0, 0, 0

    disk_parts = (commands.get("disk") or "").split()
    try:
        disk_total, disk_used = int(disk_parts[0]), int(disk_parts[1])
    except (ValueError, IndexError):
        disk_total, disk_used = 0, 0

    disk_io_parts = (commands.get("disk_io") or "").split()
    try:
        disk_read_bytes, disk_write_bytes = int(disk_io_parts[0]), int(disk_io_parts[1])
    except (ValueError, IndexError):
        disk_read_bytes, disk_write_bytes = 0, 0

    network_parts = (commands.get("network") or "").split()
    try:
        network_rx_bytes, network_tx_bytes = int(network_parts[0]), int(network_parts[1])
    except (ValueError, IndexError):
        network_rx_bytes, network_tx_bytes = 0, 0

    gpu_items = []
    gpu_memory_total = 0
    for line in (commands.get("gpu") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total = int(float(parts[1])) * 1024 * 1024
            used = int(float(parts[2])) * 1024 * 1024
            utilization = round(float(parts[3]), 1)
            temperature = round(float(parts[4]), 1)
        except ValueError:
            continue
        gpu_memory_total += total
        gpu_items.append({
            "name": parts[0],
            "memory": capacity_metric(total, used, "#7c3aed"),
            "utilization": utilization,
            "temperature": temperature,
            "temperature_percent": min(max(temperature, 0), 100),
        })

    return {
        "ok": result["ok"],
        "error": result["error"],
        "timestamp": time.time(),
        "hostname": commands.get("hostname", ""),
        "os_release": commands.get("os_release", ""),
        "kernel": commands.get("kernel", ""),
        "cpu_count": cpu_count,
        "cpu_items": cpu_items,
        "cpu_counters": cpu_counters,
        "load": {
            "values": load_values,
            "percent": load_percent,
            "style": f"width: {load_percent}%",
            "items": load_items,
        },
        "uptime": commands.get("uptime", ""),
        "memory": segmented_capacity_metric(
            memory_total,
            [
                ("已用", max(memory_used - memory_shared - memory_cache, 0), "#1667c7"),
                ("共享", memory_shared, "#8b5cf6"),
                ("缓存", memory_cache, "#f59e0b"),
                ("可用", memory_available, "#94a3b8"),
            ],
        ),
        "disk": capacity_metric(disk_total, disk_used, "#f97316"),
        "disk_io": {
            "read_bytes": disk_read_bytes,
            "write_bytes": disk_write_bytes,
            "read_total": format_bytes(disk_read_bytes),
            "write_total": format_bytes(disk_write_bytes),
        },
        "network": {
            "rx_bytes": network_rx_bytes,
            "tx_bytes": network_tx_bytes,
            "rx_total": format_bytes(network_rx_bytes),
            "tx_total": format_bytes(network_tx_bytes),
        },
        "gpu_count": len(gpu_items),
        "gpu_memory_total": format_bytes(gpu_memory_total),
        "gpus": gpu_items,
        "raw": commands,
    }


def resource_summary(resource: dict):
    metrics = remote_metrics(resource)
    system = metrics["os_release"] or metrics["kernel"] or "未知"
    system_lower = system.lower()
    if "ubuntu" in system_lower:
        os_family = "ubuntu"
    elif "debian" in system_lower:
        os_family = "debian"
    elif "centos" in system_lower:
        os_family = "centos"
    elif "rocky" in system_lower:
        os_family = "rocky"
    elif "windows" in system_lower:
        os_family = "windows"
    elif "darwin" in system_lower or "macos" in system_lower:
        os_family = "macos"
    else:
        os_family = "linux"
    return {
        "ok": metrics["ok"],
        "error": metrics["error"],
        "hostname": metrics["hostname"] or "未知",
        "system": system,
        "os_family": os_family,
        "cpu_count": metrics["cpu_count"] or 0,
        "memory_total": metrics["memory"]["total"],
        "disk_total": metrics["disk"]["total"],
        "gpu_count": metrics["gpu_count"],
        "gpu_memory_total": metrics["gpu_memory_total"],
    }


def collect_resource_summary(resource: dict):
    try:
        return resource_summary(resource)
    except Exception as error:
        return default_resource_summary(str(error))


def remote_exec(resource: dict, command: str, timeout: int = 25):
    try:
        import paramiko
    except ImportError:
        return False, "", "当前 Python 环境未安装 paramiko。"

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
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        status_code = stdout.channel.recv_exit_status()
        return status_code == 0, stdout_text, stderr_text
    except Exception as error:
        return False, "", str(error)
    finally:
        client.close()


def local_exec(command: list[str], timeout: int = 12):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as error:
        return False, "", str(error)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def local_shell(command: str, timeout: int = 12):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, shell=True)
    except Exception as error:
        return False, "", str(error)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def check_command(resource: dict | None, remote_command: str, local_command: list[str] | str, timeout: int = 12):
    if resource:
        return remote_exec(resource, remote_command, timeout=timeout)
    if isinstance(local_command, str):
        return local_shell(local_command, timeout=timeout)
    return local_exec(local_command, timeout=timeout)


def tool_check_items(resource: dict | None):
    summary = resource.get("summary", {}) if resource else {}
    server_ok = bool(summary.get("ok")) if resource else True
    server_version = ""
    if resource:
        server_version = " / ".join(
            value
            for value in (summary.get("hostname"), summary.get("system"))
            if value and value != "未知"
        )
    else:
        ok, out, err = local_shell("hostname && uname -srmo", timeout=8)
        server_ok = ok
        server_version = out or err

    checks = [
        {
            "key": "server",
            "name": "服务器版本",
            "ok": server_ok,
            "version": server_version or summary.get("error") or "未知",
            "installable": False,
            "install_label": "",
        }
    ]

    ok, out, err = check_command(
        resource,
        "command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n 1 || (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n 1)",
        "command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n 1 || (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n 1)",
    )
    checks.append({
        "key": "cuda",
        "name": "Cuda 版本",
        "ok": ok and bool(out),
        "version": out or err or "未安装",
        "installable": False,
        "install_label": "需按驱动环境安装",
    })

    ok, out, err = check_command(
        resource,
        "python3 -c \"import ultralytics; print(ultralytics.__version__)\"",
        ["python3", "-c", "import ultralytics; print(ultralytics.__version__)"],
    )
    checks.append({
        "key": "yolo",
        "name": "yolo 版本",
        "ok": ok and bool(out),
        "version": out or err or "未安装",
        "installable": True,
        "install_label": "一键安装",
    })

    ok, out, err = check_command(
        resource,
        "python3 -c \"import importlib.metadata as m; print(m.version('netkiller-yoloutils'))\"",
        ["python3", "-c", "import importlib.metadata as m; print(m.version('netkiller-yoloutils'))"],
    )
    checks.append({
        "key": "netkiller-yoloutils",
        "name": "netkiller-yoloutils",
        "ok": ok and bool(out),
        "version": out or err or "未安装",
        "installable": True,
        "install_label": "一键安装",
    })
    return checks


def install_tool(resource: dict | None, tool: str):
    commands = {
        "yolo": "python3 -m pip install -U ultralytics",
        "netkiller-yoloutils": "python3 -m pip install -U netkiller-yoloutils -i https://pypi.tuna.tsinghua.edu.cn/simple",
    }
    command = commands.get(tool)
    if not command:
        return False, f"{tool} 不支持一键安装"
    if resource:
        ok, out, err = remote_exec(resource, command, timeout=180)
    else:
        ok, out, err = local_shell(command, timeout=180)
    return ok, out or err or ("安装完成" if ok else "安装失败")


@router.get("/resources/check")
@router.get("/resources/{project}/check")
def resources_check(request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    resource_id = request.query_params.get("server", "")
    resource = find_resource(workspace, resource_id) if resource_id else None
    checks = tool_check_items(resource)
    response = templates.TemplateResponse(
        request=request,
        name="resources/check.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resources": resource_list_items(workspace),
            "resource": resource,
            "checks": checks,
            "message": request.query_params.get("message", ""),
            "error": request.query_params.get("error", ""),
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.post("/resources/check/install")
@router.post("/resources/{project}/check/install")
async def resources_check_install(request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    tool = (form.get("tool", [""])[0] or "").strip()
    resource_id = (form.get("server", [""])[0] or "").strip()
    resource = find_resource(workspace, resource_id) if resource_id else None
    ok, output = install_tool(resource, tool)
    params = {}
    if resource_id:
        params["server"] = resource_id
    params["message" if ok else "error"] = output[:160]
    return RedirectResponse(
        url=f"{resources_base(current_project)}/check?{urlencode(params)}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/resources")
@router.get("/resources/{project}")
def resources(request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    legacy_project = request.query_params.get("project", "")
    if not project and legacy_project and project_dir(workspace, legacy_project):
        return RedirectResponse(url=f"/resources/{legacy_project}", status_code=status.HTTP_303_SEE_OTHER)
    current_project = current_project_from_request(request, workspace, project)
    response = templates.TemplateResponse(
        request=request,
        name="resources/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resources": resource_list_items(workspace),
            "compute_config": compute_config(workspace),
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/resources/server/{resource_id}")
@router.get("/resources/{project}/server/{resource_id}")
def resource_detail(resource_id: str, request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    resource = find_resource(workspace, resource_id)
    if resource is None:
        return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="resources/detail.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resource": resource,
            "metrics": remote_metrics(resource),
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/resources/server/{resource_id}/metrics")
@router.get("/resources/{project}/server/{resource_id}/metrics")
def resource_metrics(resource_id: str, request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    resource = find_resource(workspace, resource_id)
    if resource is None:
        return JSONResponse({"ok": False, "error": "服务器不存在"}, status_code=404)
    return JSONResponse(remote_metrics(resource))


@router.get("/resources/server/{resource_id}/ssh")
@router.get("/resources/{project}/server/{resource_id}/ssh")
def resource_ssh(resource_id: str, request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    resource = find_resource(workspace, resource_id)
    if resource is None:
        return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="resources/ssh.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resource": resource,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


def sftp_items(resource: dict, requested_path: str):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = ssh_connect_kwargs(resource)
    client.connect(
        hostname=resource["host"],
        port=resource["port"],
        username=resource["username"],
        timeout=8,
        banner_timeout=8,
        auth_timeout=8,
        look_for_keys=False,
        allow_agent=False,
        **connect_kwargs,
    )
    sftp = None
    try:
        sftp = client.open_sftp()
        path = (requested_path or ".").strip() or "."
        sftp.chdir(path)
        current_path = sftp.normalize(".")
        rows = []
        for entry in sorted(sftp.listdir_attr("."), key=lambda item: (not stat_module.S_ISDIR(item.st_mode), item.filename.lower())):
            is_dir = stat_module.S_ISDIR(entry.st_mode)
            child_path = posixpath.join(current_path, entry.filename)
            rows.append(
                {
                    "name": entry.filename,
                    "path": child_path,
                    "href": "?" + urlencode({"path": child_path}),
                    "type": "目录" if is_dir else "文件",
                    "is_dir": is_dir,
                    "size": entry.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.st_mtime)),
                }
            )
        parent_path = posixpath.dirname(current_path.rstrip("/")) or "/"
        return {
            "ok": True,
            "path": current_path,
            "parent_href": "?" + urlencode({"path": parent_path}),
            "items": rows,
            "error": "",
        }
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        client.close()


@router.get("/resources/server/{resource_id}/sftp")
@router.get("/resources/{project}/server/{resource_id}/sftp")
def resource_sftp(resource_id: str, request: Request, project: str = "", path: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    resource = find_resource(workspace, resource_id)
    if resource is None:
        return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)
    try:
        listing = sftp_items(resource, path)
    except Exception as error:
        listing = {"ok": False, "path": path or ".", "parent_href": "", "items": [], "error": str(error)}
    response = templates.TemplateResponse(
        request=request,
        name="resources/sftp.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resource": resource,
            "listing": listing,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


def open_ssh_shell(resource: dict):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = ssh_connect_kwargs(resource)
    client.connect(
        hostname=resource["host"],
        port=resource["port"],
        username=resource["username"],
        timeout=8,
        banner_timeout=8,
        auth_timeout=8,
        look_for_keys=False,
        allow_agent=False,
        **connect_kwargs,
    )
    channel = client.invoke_shell(term="xterm-256color", width=120, height=32)
    channel.setblocking(False)
    channel.send("stty erase '^?' 2>/dev/null; bind '\"\\e[3~\": delete-char' 2>/dev/null\r")
    return client, channel


@router.websocket("/resources/server/{resource_id}/ssh/ws")
@router.websocket("/resources/{project}/server/{resource_id}/ssh/ws")
async def resource_ssh_ws(websocket: WebSocket, resource_id: str, project: str = ""):
    await websocket.accept()
    workspace = workspace_path()
    if team_mode_enabled() and not current_username(websocket, workspace):
        await websocket.send_text("\r\n请先登录团队账号\r\n")
        await websocket.close(code=1008)
        return
    resource = find_resource(workspace, resource_id)
    if resource is None:
        await websocket.send_text("\r\n服务器不存在\r\n")
        await websocket.close()
        return
    try:
        client, channel = await asyncio.to_thread(open_ssh_shell, resource)
    except Exception as error:
        await websocket.send_text(f"\r\nSSH 连接失败：{error}\r\n")
        await websocket.close()
        return

    async def reader():
        try:
            while True:
                if channel.closed:
                    break
                if channel.recv_ready():
                    data = channel.recv(4096).decode("utf-8", errors="replace")
                    await websocket.send_text(data)
                else:
                    await asyncio.sleep(0.03)
        except Exception:
            pass

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            text = await websocket.receive_text()
            if text:
                channel.send(text)
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            channel.close()
        finally:
            client.close()


@router.get("/resources/server/{resource_id}/edit")
@router.get("/resources/{project}/server/{resource_id}/edit")
def resource_edit(resource_id: str, request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    resource = find_resource(workspace, resource_id)
    if resource is None:
        return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="resources/edit.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "resources",
            "current_project": current_project,
            "resources_base": resources_base(current_project),
            "resource": resource,
            **header_context(request, workspace),
        },
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.post("/resources/server/{resource_id}/edit")
@router.post("/resources/{project}/server/{resource_id}/edit")
async def update_resource(resource_id: str, request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    updated = resource_form_data(form)
    updated["summary"] = collect_resource_summary(updated)
    items = read_resources(workspace)
    next_items = []
    found = False
    for item in items:
        if item["id"] == resource_id:
            next_items.append({"id": resource_id, **updated})
            found = True
        else:
            next_items.append(item)
    if not found:
        return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)
    write_resources(workspace, next_items)
    return RedirectResponse(url=resource_detail_url(current_project, resource_id), status_code=status.HTTP_303_SEE_OTHER)


@router.post("/resources")
@router.post("/resources/{project}")
async def add_resource(request: Request, project: str = ""):
    workspace = workspace_path()
    login_response = require_team_login(request, workspace)
    if login_response:
        return login_response
    current_project = current_project_from_request(request, workspace, project)
    form = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    payload = resource_form_data(form)
    payload["summary"] = collect_resource_summary(payload)
    items = read_resources(workspace)
    items.append(
        {
            "id": resource_key(payload),
            **payload,
        }
    )
    write_resources(workspace, items)
    return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)

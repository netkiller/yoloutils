import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.project import header_context, project_dir, require_team_login, workspace_path


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
        "note": str(item.get("note") or ""),
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


def resource_list_items(workspace: Path):
    items = read_resources(workspace)
    if not items:
        return []

    summaries = {}
    with ThreadPoolExecutor(max_workers=min(4, len(items))) as executor:
        futures = {executor.submit(resource_summary, item): item["id"] for item in items}
        for future in as_completed(futures):
            resource_id = futures[future]
            try:
                summaries[resource_id] = future.result()
            except Exception as error:
                summaries[resource_id] = {
                    "ok": False,
                    "error": str(error),
                    "hostname": "未知",
                    "system": "未知",
                    "os_family": "linux",
                    "cpu_count": "未知",
                    "memory_total": "未知",
                    "disk_total": "未知",
                    "gpu_count": 0,
                    "gpu_memory_total": "0 B",
                }
    return [{**item, "summary": summaries.get(item["id"], {})} for item in items]


def resource_form_data(form: dict):
    name = (form.get("name", [""])[0] or "").strip()
    host = (form.get("host", [""])[0] or "").strip()
    username = (form.get("username", [""])[0] or "").strip()
    password = (form.get("password", [""])[0] or "").strip()
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
        "note": note,
    }


def resource_detail_url(current_project: str, resource_id: str):
    return f"{resources_base(current_project)}/server/{resource_id}"


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
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            password=resource["password"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
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
            "memory": "free -b | awk '/Mem:/ {print $2\" \"$3\" \"$7}'",
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
        memory_total, memory_used = int(memory_parts[0]), int(memory_parts[1])
    except (ValueError, IndexError):
        memory_total, memory_used = 0, 0

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
        },
        "uptime": commands.get("uptime", ""),
        "memory": capacity_metric(memory_total, memory_used, "#16a34a"),
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
    items = read_resources(workspace)
    items.append(
        {
            "id": resource_key(payload),
            **payload,
        }
    )
    write_resources(workspace, items)
    return RedirectResponse(url=resources_base(current_project), status_code=status.HTTP_303_SEE_OTHER)

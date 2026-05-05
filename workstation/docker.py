#!/usr/bin/env python3.13
##################################################
# Home	: https://www.netkiller.cn
# Author: Neo <netkiller@msn.com>
# Upgrade: 2025-04-25
# yolo workstation docker compose
##################################################

##################################################
import sys

try:
    # sys.path.insert(0, ".")
    # sys.path.insert(1, "/Users/neo/workspace/devops")
    sys.path.insert(0, "/Users/neo/GitHub/devops")
    # sys.path.insert(3, "/srv/devops")
    from netkiller.docker import *
except ModuleNotFoundError as err:
    print("pip install netkiller-devops, %s" % (err))
    exit()

development = Composes("development")
# testing = Composes("testing")
# production = Composes("production")

volumes = Volumes()
# volumes.add("postgres")
# volumes.add("caddy", name="caddy")
# volumes.add("redis", name="redis")
volumes.add("site-packages", name="site-packages")
development.volumes(volumes)

# -----------------------------------------------------------------------------------------------
# Caddy
# -----------------------------------------------------------------------------------------------
caddy = Services("caddy")
caddy.image("caddy:latest")
caddy.container_name("caddy")
caddy.restart("unless-stopped")
# service.hostname('www.netkiller.cn')
# nginx.extra_hosts(extra_hosts)
# service.extra_hosts(['db.netkiller.cn:127.0.0.1','cache.netkiller.cn:127.0.0.1','api.netkiller.cn:127.0.0.1'])
caddy.volumes(
    [
        # "/etc/letsencrypt:/etc/letsencrypt",
        "/opt/caddy/Caddyfile:/etc/caddy/Caddyfile:ro",
        "/opt/caddy/Caddyfile.d:/etc/caddy/Caddyfile.d:ro",
        "/opt/caddy/data:/data",
    ]
)

caddy.environment(["TZ=Asia/Shanghai"])
caddy.ports(["80:80", "443:443", "443:443/udp"])

caddy.file(
    "/opt/caddy/Caddyfile",
    # "/etc/caddy/Caddyfile.d/vhost.caddyfile",
    """
{
    email netkiller@msn.com
    # 日志
    log {
        output stdout
        format console
    }
}

www.netkiller.cn {
    root * /data/www
    file_server
    try_files {path} {path}/ /index.html
}

api.netkiller.cn {

    reverse_proxy http://api:8080 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        transport http {
            read_timeout 120s
            write_timeout 120s
        }
    }

    # respond /status "Caddy 2 running" 200
}

demo.netkiller.cn {
    header {
		Cache-Control "no-store, no-cache, must-revalidate, max-age=0"
		Pragma "no-cache"
		Expires "0"
		-Etag
		-Last-Modified
	}
    root * /data/demo
    file_server
    try_files {path} {path}/ /index.html
}

yolo.netkiller.cn {
    reverse_proxy http://admin:8080
}
""",
)

# development.services(caddy)
# ------------------------------------------------------------
# Aiot Gateway
# ------------------------------------------------------------
# docker run --rm -v site-packages:/usr/local/lib/python3.14/site-packages -v /opt/gateway:/app python:3.14 pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# docker run --rm -v site-packages:/data python:3.14 ls -l /data
# ------------------------------------------------------------
yolo = Services("yolo")
yolo.container_name("yolo")
yolo.image("python:3.14").restart("unless-stopped")
yolo.environment({"TZ": "Asia/Shanghai",
                     "DATABASE_URL": "postgresql+psycopg2://dev:2BEA35C869A0@db.leeclaws.com:5432/dev",
                     "MCP_PORT": "8080"
                     })
yolo.volumes(["site-packages:/usr/local/lib/python3.14/site-packages", "/opt/gateway:/app:ro"])
# gateway.ports(["8001:8080"])
yolo.working_dir("/app")
yolo.entrypoint(["sh",
                    "entrypoint"]).command(["-w /opt/workspace","--team","--demo"])
development.services(yolo)

# development.workdir()
# development.dump()
# production.dump()

if __name__ == "__main__":
    try:
        docker = Docker()
        # docker.none()
        # docker.env({'DOCKER_HOST':'ssh://root@192.168.30.13','COMPOSE_PROJECT_NAME':'experiment'})
        # docker.sysctl({"vm.overcommit_memory": "1"})
        docker.environment(development)
        # docker.environment(testing)
        # docker.environment(production)
        docker.main()
    except KeyboardInterrupt:
        print("Crtl+C Pressed. Shutting down.")

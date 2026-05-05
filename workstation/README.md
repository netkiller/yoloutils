# Yolo workstation

## 本地运行

本地 FastAPI 站点可以使用 `start.py` 启动：

```shell
cd workstation
python3 start.py -w /Users/neo/tmp/yolo/source
```

常用参数：

```shell
python3 start.py -w /Users/neo/tmp/yolo/source -p 8000 --host 0.0.0.0
python3 start.py -w /Users/neo/tmp/yolo/source -s /path/to/dataset -r /path/to/run -c /path/to/classes.txt
python3 start.py -w /Users/neo/tmp/yolo/source --open -t --mDNS netkiller.local
python3 start.py -w /Users/neo/tmp/yolo/source --reload
```

也可以使用 `run.sh` 透传参数：

```shell
./entrypoint.sh -w /Users/neo/tmp/yolo/source --reload
```

启动后可访问：

- 标注：`http://127.0.0.1:8000/annotate/`
- 数据集：`http://127.0.0.1:8000/dataset`
- 训练：`http://127.0.0.1:8000/train`

后台运行：

```shell
python3 start.py -w /Users/neo/tmp/yolo/source -d
```

后台模式会在工作目录写入 `.yoloutils-workstation.pid` 和 `.yoloutils-workstation.log`。

## Docker 部署

```shell

```
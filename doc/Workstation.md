# Yolo Workstation

Yolo Workstation 是 `yoloutils` 内置的 YOLO 图像管理工作站。命令行启动后会运行一个本地 FastAPI 站点，用于浏览 `--workspace` 目录中的图片、`classes.txt` 和同名 YOLO `.txt` 标注文件。

## 启动方式

前台运行：

```shell
python src/netkiller/yoloutils/yoloutils.py workstation -w /Users/neo/tmp/yolo/source
```

后台运行：

```shell
python src/netkiller/yoloutils/yoloutils.py workstation -w /Users/neo/tmp/yolo/source -d
```

可选参数：

```shell
python src/netkiller/yoloutils/yoloutils.py workstation \
  --workspace /Users/neo/tmp/yolo/source \
  --host 127.0.0.1 \
  --port 8000 \
  --daemon
```

说明：

- `-w, --workspace`：YOLO 数据集工作目录。
- `--host`：监听地址，默认 `127.0.0.1`。
- `-p, --port`：监听端口，默认 `8000`。
- `-d, --daemon`：后台运行。
- 后台运行会在工作目录写入 `.yoloutils-workstation.pid` 和 `.yoloutils-workstation.log`。

## 数据扫描

进入系统后会遍历 `--workspace`：

- 查找第一个 `classes.txt`，作为标签名称来源。
- 递归识别图片文件，扩展名使用 `Common.image_exts`。
- 图片同目录同名 `.txt` 作为 YOLO 标注文件。
- YOLO 标注格式为：`class_id cx cy width height`。

## 页面布局

页面分为四个主要区域：

- 左侧目录栏：展示 `--workspace` 下的目录树。
- 文件栏：展示当前目录中的图片文件，并标识标注状态。
- 图像栏：展示当前图片；存在有效标注时绘制 YOLO box。
- 右侧信息栏：展示标签、图片直方图和 EXIF 信息。

底部 footer：

- 左侧显示 `位置：--workspace 路径`。
- 右侧显示统计项：`图像`、`.txt`、`损坏图像`、`无效 .txt`。

## 分栏交互

页面支持多处分栏拖动：

- 目录栏和文件栏之间可左右拖动调整比例。
- 文件栏和图像栏之间可左右拖动调整比例。
- 标签和直方图之间可上下拖动调整比例。
- 标签/直方图区域和 EXIF 区域之间可上下拖动调整比例。

隐藏和折叠：

- 目录栏可隐藏；隐藏后释放的空间由图像栏填充。
- 右侧信息栏可隐藏；隐藏后释放的空间由图像栏填充。
- 直方图可折叠；折叠后标签高度保持不变，释放的空间由 EXIF 区域接管。
- EXIF 可折叠到底部，只保留标题栏。

目录栏和文件栏标题已固定：

- 滚动目录树时，`目录` 标题不滚动。
- 滚动文件列表时，`文件` 标题不滚动。

文件列表状态颜色：

- 已标注且 `.txt` 有效的图片显示为绿色。
- 空 `.txt`、无效 `.txt` 或损坏图片显示为红色。
- 未标注图片保持默认颜色。

## 图像栏工具

图像栏头部提供工具按钮：

- `自动`：自动标注激活开关。目前实现为前端激活状态，后续可接入模型推理。
- `删除`：清空当前图片的标注框。
- `重置`：重新从同名 `.txt` 读取当前图片标注。
- `保存`：把当前标注写回同名 `.txt`。

保存时会调用后端接口写入 YOLO txt，格式仍为：

```text
class_id cx cy width height
```

坐标会按 6 位小数保存。

## 右侧信息栏

标签：

- 来源于 `classes.txt`。
- 每个标签显示索引、名称和颜色块。
- 图像中的 box 颜色与标签索引对应。

直方图：

- 展示当前图片 RGB 三通道直方图。
- 直方图在前端基于当前图片像素生成。
- 拖动面板比例或展开直方图时会重新绘制尺寸。

EXIF：

- 展示当前图片文件信息。
- 展示 PIL 可读取到的 EXIF 标签。
- 如果图片没有 EXIF，则显示“没有 EXIF 信息”。

## 统计口径

`/api/statistics` 返回工作站底部 footer 使用的统计数据。

字段：

- `workspace`：当前工作目录绝对路径。
- `images`：识别到的图片总数。
- `images_damaged`：PIL 打开或校验失败的图片数量。
- `txt_total`：存在同名 `.txt` 的图片数量。
- `txt_missing`：缺少同名 `.txt` 的图片数量。
- `txt_empty`：同名 `.txt` 存在但内容为空的数量。
- `txt_invalid`：同名 `.txt` 内容格式无效的数量。
- `txt_valid`：同名 `.txt` 内容有效的数量。
- `txt_invalid_total`：`txt_empty + txt_invalid`。
- `classes`：`classes.txt` 中的标签数量。

无效 `.txt` 判定：

- 非 UTF-8 或读取失败。
- 文件内容为空。
- 任意行不是 5 列。
- `class_id` 或坐标无法解析为数值。
- `class_id` 小于 0。
- `class_id` 超出 `classes.txt` 标签数量范围。

## 后端接口

当前 FastAPI 接口：

- `GET /`：工作站页面。
- `GET /api/tree`：目录树。
- `GET /api/files?directory=<path>`：当前目录图片列表。
- `GET /api/classes`：标签列表和 `classes.txt` 路径。
- `GET /api/statistics`：底部统计数据。
- `GET /api/annotation?path=<image>`：读取图片同名 YOLO 标注。
- `POST /api/annotation`：保存图片 YOLO 标注。
- `GET /api/exif?path=<image>`：读取图片文件信息和 EXIF。
- `GET /media?path=<image>`：读取图片文件。

`POST /api/annotation` 请求示例：

```json
{
  "path": "images/example.jpg",
  "boxes": [
    {
      "class_id": 0,
      "cx": 0.5,
      "cy": 0.5,
      "width": 0.2,
      "height": 0.3
    }
  ]
}
```

## 路径安全

所有接口中的 `path` 都会经过工作目录约束：

- 只能访问 `--workspace` 内部文件。
- 不能通过 `..` 跳出工作目录。
- `/media` 只允许返回识别为图片的文件。

## 当前限制

- `自动` 按钮目前只实现激活状态，尚未接入模型自动推理。
- 当前页面支持删除、重置和保存已有标注数据，但还没有实现鼠标绘制新 box、拖动调整 box 或修改类别。
- 损坏图像统计依赖 PIL `verify()`，对部分浏览器可显示但 PIL 不支持的格式可能会被统计为损坏。

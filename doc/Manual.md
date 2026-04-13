# yoloutils 操作手册

`yoloutils` 是一组面向 YOLO 数据集的命令行工具，用于标签统计、标签筛选、索引替换、数据集整理、图片裁剪、分类数据拆分和模型批量测试。

作者: Neo <netkiller@msn.com>  
官网: https://www.netkiller.cn

---

## 1. 工具概览

当前源码中的子命令如下：

| 子命令 | 作用 |
|---|---|
| `label` | 查看 `classes.txt`、统计标签数量、搜索标签所在文件 |
| `merge` | 按文件名合并两组 YOLO 标注 |
| `copy` | 按标签名称复制标注文件和图片 |
| `remove` | 删除指定标签索引或标签名称 |
| `change` | 批量替换标签索引 |
| `crop` | 使用模型或现有 `txt` 标注裁剪图片 |
| `labelimg` | 手工整理 labelimg 数据并生成 YOLO 训练目录 |
| `auto` | 用现有模型自动生成 YOLO 标签 |
| `resize` | 按长边尺寸缩放图片 |
| `classify` | 处理分类数据集并划分 `train/test/val` |
| `test` | 单模型批量推理并输出表格或 CSV |
| `diff` | 多模型并发对比并输出表格或 CSV |

查看总帮助：

```shell
yoloutils -h
```

---

## 2. 安装与运行

### 2.1 环境要求

- Python `>= 3.13`
- 依赖项：
  - `pillow`
  - `opencv-python`
  - `pyyaml`
  - `tqdm`
  - `texttable`
  - `ultralytics`

### 2.2 从 PyPI 安装

```shell
pip install netkiller-yoloutils
```

### 2.3 从源码安装

```shell
git clone https://github.com/netkiller/yoloutils.git
cd yoloutils
pip install -e .
```

### 2.4 构建 wheel 安装

```shell
pip install build
python -m build
pip install dist/netkiller_yoloutils-*.whl --force-reinstall
```

### 2.5 查看某个子命令帮助

```shell
yoloutils label -h
yoloutils crop -h
yoloutils auto -h
yoloutils classify -h
yoloutils test -h
yoloutils diff -h
```

---

## 3. 数据准备

### 3.1 YOLO 检测数据目录

大多数命令面向 YOLO 检测标注，目录通常如下：

```text
dataset/
├── classes.txt
├── image001.jpg
├── image001.txt
├── image002.jpg
├── image002.txt
└── subdir/
    ├── image101.jpg
    └── image101.txt
```

说明：

- `classes.txt` 为类别清单，每行一个类别名称。
- 标注文件为 YOLO 标准格式：`class x_center y_center width height`。
- 当前源码中大多数流程按 `.jpg + .txt` 成对处理，`png` 支持并不完整，实际使用时建议统一为 `.jpg`。
- 多个命令会递归扫描 `source` 目录下的所有标注文件。

### 3.2 分类数据目录

`classify` 命令要求按类别分目录：

```text
source/
├── cat/
│   ├── 001.jpg
│   └── 002.jpg
├── dog/
│   ├── 101.jpg
│   └── 102.jpg
└── bird/
    └── 201.jpg
```

### 3.3 日志文件

程序运行后会在当前工作目录生成类似下面的日志文件：

```text
yoloutils.2026-04-07.log
```

---

## 4. 命令详解

### 4.1 `label`

用于查看 `classes.txt`、统计标签数量以及检索包含某些标签的标注文件。

命令格式：

```shell
yoloutils label --source DATASET [--classes | --total | --index | --search 0 1 2]
```

帮助信息：

```shell
usage: yoloutils.py label [-h] [--source /tmp/dir1] [--classes] [--total]
                          [--index] [--search 1 2 3 [1 2 3 ...]]

options:
  -h, --help            show this help message and exit
  --source /tmp/dir1    目录
  --classes             查看 classes.txt 文件
  --total               统计标签图数量
  --index               统计标签索引数量
  --search 1 2 3 [1 2 3 ...]
                        搜索标签
```

详细使用说明（推荐流程）：

1. 准备数据目录：确保 `--source` 目录存在，且包含 `classes.txt` 与若干 `*.txt` 标注文件。
2. 先执行 `--classes`：确认标签索引和标签名称对应关系，避免后续按索引分析时看错类别。
3. 按目标选择统计方式：
   - 看标签总量用 `--total`；
   - 看纯索引计数用 `--index`；
   - 查特定索引落在哪些文件用 `--search`。
4. 核验结果：随机打开 `--search` 输出的若干文件，确认首列索引与表格统计一致。

常用示例：

```shell
# 查看 classes.txt
yoloutils label --source ./dataset --classes

# 统计各标签出现次数
yoloutils label --source ./dataset --total

# 仅按索引统计
yoloutils label --source ./dataset --index

# 查找包含索引 0 和 2 的标注文件
yoloutils label --source ./dataset --search 0 2
```

实现说明：

- `--classes` 读取 `source/classes.txt` 并以表格输出。
- `--total` 统计的是标注框数量，不是图片数量。
- `--index` 与 `--total` 使用同一套统计逻辑，只是不再映射成标签名称。
- `--search` 返回包含指定索引的标注文件路径列表。

### 4.2 `merge`

将左侧目录中的标注文件与右侧目录中同名标注合并到输出目录，同时复制左侧目录中的图片。

命令格式：

```shell
yoloutils merge \
    --left ./left \
    --right ./right \
    --output ./merged \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py merge [-h] [--left /tmp/dir1] [--right /tmp/dir2]
                          [--output /tmp/output] [--clean]

options:
  -h, --help            show this help message and exit
  --left /tmp/dir1      左侧目录
  --right /tmp/dir2     右侧目录
  --output /tmp/output  最终输出目录
  --clean               清理之前的数据
```

详细使用说明（推荐流程）：

1. 准备左右目录：`left` 和 `right` 都应包含 YOLO 标签文件，且两边使用可匹配的文件名规则。
2. 先在小样本目录验证：建议先拷贝 5-10 组样本到临时目录验证合并规则，再全量执行。
3. 正式执行时指定 `--output`，若需要覆盖旧结果再加 `--clean`。
4. 核验输出：检查 `output` 中图片是否来自左侧目录，标签文件是否为“左+右”拼接结果。

使用前提：

- 左右目录中都使用 YOLO `.txt` 标注文件。
- 图片从 `--left` 目录复制到输出目录。
- 当前实现默认图片扩展名为 `.jpg`。
- 右侧文件名匹配规则较特殊：左侧若为 `name_0.txt`，右侧会查找 `name.txt`。

适用场景示例：

```text
left/
├── sample_0.jpg
└── sample_0.txt

right/
└── sample.txt
```

合并后：

```text
merged/
├── sample_0.jpg
└── sample_0.txt
```

实现说明：

- 如果右侧缺少对应标注，则仅复制左侧标注。
- 如果右侧存在对应标注，则把两个文本内容直接拼接写入输出文件。
- 当前源码已修正顶层 CLI 的 `merge` 分发，命令可以正常进入合并逻辑。

### 4.3 `copy`

按标签名称筛选标注文件，并复制对应的标注和图片到目标目录。

命令格式：

```shell
yoloutils copy \
    --source ./dataset \
    --target ./picked \
    --label person,dog \
    --uuid \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py copy [-h] [--source SOURCE] [--target TARGET]
                         [--label LABEL] [-u] [-c]

options:
  -h, --help       show this help message and exit
  --source SOURCE  图片来源地址
  --target TARGET  图片目标地址
  --label LABEL    逗号分割多个标签
  -u, --uuid       UUID 文件名
  -c, --clean      清理目标文件夹
```

详细使用说明（推荐流程）：

1. 先确认 `source/classes.txt` 标签名称，`--label` 必须和文件里的名称完全一致。
2. 若只提取某些类别，传入 `--label person,dog`；若不传 `--label`，按源码行为会复制全部标签文件。
3. 如需避免重名覆盖，建议加 `--uuid`；如需覆盖旧结果，加 `--clean`。
4. 核验结果：在 `target` 中检查是否同时有同名 `jpg/txt`，并确认 `classes.txt` 已复制。

常用示例：

```shell
# 复制含有 cat 标签的样本
yoloutils copy --source ./dataset --target ./picked --label cat

# 复制多个标签
yoloutils copy --source ./dataset --target ./picked --label person,car,bicycle

# 输出文件名改为 UUID
yoloutils copy --source ./dataset --target ./picked --label dog --uuid
```

实现说明：

- 复制前会读取 `source/classes.txt`，把标签名称转换为索引。
- `--label` 采用逗号分隔，如 `person,dog,car`。
- 会把 `classes.txt` 一并复制到目标目录。
- 当前实现主要按 `.jpg` 配对复制图片。
- 该命令的复制逻辑依赖文件名和目录状态，建议先在小样本目录验证输出结果。

### 4.4 `remove`

从 YOLO 标注文件中删除指定标签。可以按索引删除，也可以按标签名称删除。

命令格式：

```shell
yoloutils remove \
    --source ./dataset \
    --target ./cleaned \
    --classes 0 3 5 \
    --clean
```

或：

```shell
yoloutils remove \
    --source ./dataset \
    --label cat dog
```

帮助信息：

```shell
usage: yoloutils.py remove [-h] [--source SOURCE] [--target TARGET] [--clean]
                           [--classes 1 2 3 [1 2 3 ...]]
                           [--label label1 label2 [label1 label2 ...]]

options:
  -h, --help            show this help message and exit
  --classes 1 2 3 [1 2 3 ...]
                        标签序号
  --label label1 label2 [label1 label2 ...]
                        标签名称

通用参数:
  --source SOURCE       图片来源地址
  --target TARGET       图片目标地址
  --clean               清理之前的数据
```

详细使用说明（推荐流程）：

1. 明确删除方式：
   - 已知索引用 `--classes 0 3 5`；
   - 已知标签名用 `--label cat dog`（会自动从 `classes.txt` 映射索引）。
2. 先做安全试跑：优先指定 `--target` 输出新目录，不建议第一次就原地修改。
3. 确认结果目录：有变更的标签文件会写入目标目录；删除后为空的文件会连同对应图片一起移除。
4. 核验：执行后用 `yoloutils label --source <target> --index` 复查删除是否生效。

实现说明：

- `--classes` 接收标签索引列表。
- `--label` 先读取 `source/classes.txt`，再转换成索引列表。
- 如果指定了 `--target`，会把结果写入目标目录；如果不指定，则直接原地修改源文件。
- 某个标注文件在删除后若没有任何标注行，程序会删除对应的 `.txt` 和同名 `.jpg`。
- 当前实现写入 `--target` 时只保留文件名，不保留原始子目录结构；若不同子目录存在同名文件，可能发生覆盖。

常用示例：

```shell
# 删除索引 0 和 1
yoloutils remove --source ./dataset --target ./cleaned --classes 0 1

# 删除标签名称 cat、dog
yoloutils remove --source ./dataset --target ./cleaned --label cat dog

# 直接原地删除
yoloutils remove --source ./dataset --classes 7
```

### 4.5 `change`

批量修改标签索引，直接原地覆盖源标注文件。

命令格式：

```shell
yoloutils change \
    --source ./dataset \
    --search 0 1 2 \
    --replace 3 4 5
```

帮助信息：

```shell
usage: yoloutils.py change [-h] [--source /tmp/dir1]
                           [--search 1 2 3 [1 2 3 ...]]
                           [--replace 4 5 6 [4 5 6 ...]]

options:
  -h, --help            show this help message and exit
  --source /tmp/dir1    目录
  --search 1 2 3 [1 2 3 ...]
                        标签序号
  --replace 4 5 6 [4 5 6 ...]
                        标签名称
```

详细使用说明（推荐流程）：

1. 先确定映射关系：`--search` 与 `--replace` 按位置一一对应，例如 `0->5, 1->6`。
2. 先备份源目录：该命令是原地改写，不提供 `--target`。
3. 执行替换后，立即用 `yoloutils label --source <dir> --index` 对比前后计数变化。
4. 若替换关系较多，建议先在小目录验证，确认没有错位映射再跑全量。

实现说明：

- `--search` 与 `--replace` 按位置一一对应。
- 会递归处理 `source` 下所有 `.txt`，跳过 `classes.txt`。
- 该命令没有 `--target` 输出目录，修改直接写回原文件。

常用示例：

```shell
# 把 0 -> 5, 1 -> 6
yoloutils change --source ./dataset --search 0 1 --replace 5 6
```

### 4.6 `crop`

`crop` 支持两种裁切模式：

- 模型模式：`--model`，先检测再裁切。
- TXT 模式：`--txt --imgsz`，按同名标注框裁切。

命令格式（模型模式）：

```shell
yoloutils crop \
    --source ./images \
    --target ./cropped \
    --model ./best.pt \
    --output ./predict \
    --clean
```

命令格式（TXT 模式）：

```shell
yoloutils crop \
    --source ./images \
    --target ./cropped \
    --txt \
    --imgsz 640 \
    --csv ./missed.csv \
    --clean
```

帮助信息（当前源码）：

```shell
usage: yoloutils.py crop [-h] [--source SOURCE] [--target TARGET] [--clean]
                         [-c result.csv] [--model best.pt]
                         [--output /tmp/output] [--txt] [--imgsz 640]

options:
  -h, --help            show this help message and exit
  --source SOURCE       图片来源地址
  --target TARGET       图片目标地址
  --clean               清理之前的数据
  -c, --csv result.csv  未处理文件列表

基于模型裁切:
  用指定模型识别后，将 box 框内的图像保存指定目录

  --model best.pt       模型
  --output /tmp/output  Yolo 输出目录

基于txt标准裁切:
  用.txt文件中的box框为基准，向外扩展裁切

  --txt                 自动标注
  --imgsz 640           矩形长边
```

详细使用说明（推荐流程）：

1. 先选模式：`--model` 或 `--txt`，至少需要一个。
2. 指定输入输出：`--source`、`--target` 必填。
3. 需要清理旧数据时加 `--clean`。
4. 如需导出“未处理文件”清单，加 `--csv <file>`。
5. 核验：
   - 模型模式：检查 `target` 与可选 `output` 的结果图。
   - TXT 模式：检查裁切后 `jpg/txt` 是否同名、框是否位于裁切图有效区域。

实现说明：

- 输入扫描：递归读取 `source/**/*`，并按 `Common.image_exts` 过滤（`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`）。
- `--target` 保留原始相对目录结构。
- 模型模式：`--model` 必填；`--output` 会额外保存带框图和 `output/crop/`。
- TXT 模式：
  - 要求图片存在同名 `.txt`，缺失会记入“未处理”。
  - 先读取首个框判断：若框宽或框高大于 `imgsz`，不裁切，直接复制原图和原标签。
  - 若框尺寸在 `imgsz` 内，则以该框中心裁切 `imgsz x imgsz`，并重算/裁剪所有标签框坐标。
- 终端会打印“未处理文件”ASCII 表（有未处理时）和一行统计摘要。
- 传入 `--csv` 时，会把“未处理文件”表格行写入 CSV（首行为 `未处理文件`）。

常用示例：

```shell
# 模型模式
yoloutils crop --source ./images --target ./cropped --model ./best.pt

# TXT 模式（长边 640）
yoloutils crop --source ./images --target ./cropped --txt --imgsz 640

# TXT 模式并导出未处理文件
yoloutils crop --source ./images --target ./cropped --txt --imgsz 640 --csv ./missed.csv
```

### 4.7 `labelimg`

`labelimg` 负责把现有 `labelimg` 标注数据整理成 YOLO 训练目录结构，并生成 `data.yaml`。

命令格式：

```shell
yoloutils labelimg \
    --source ./labelimg_data \
    --target ./yolo_data \
    --val 10 \
    --uuid \
    --clean
```

帮助信息（当前源码）：

```shell
usage: yoloutils.py labelimg [-h] [--source SOURCE] [--target TARGET] [--clean] [--classes CLASSES] [--val 10] [--uuid] [--check]

options:
  -h, --help         show this help message and exit
  --source SOURCE    图片来源地址
  --target TARGET    图片目标地址
  --clean            清理之前的数据
  --classes CLASSES  classes.txt 文件
  --val 10           检验数量
  --uuid             输出文件名使用UUID
  --check            图片检查 corrupt JPEG restored and saved
```

详细使用说明（推荐流程）：

1. 准备 `source/classes.txt` 与成对 `图片+txt`。
2. 先在小样本目录验证，再全量执行。
3. 首次建议加 `--clean`，保证输出目录可复现。
4. 运行后核对 `data.yaml`、`train/val` 目录结构和标签文件数量。

输出目录结构：

```text
yolo_data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

实现说明：

- 实际读取 `source/classes.txt`。
- `--val` 表示每个标签抽样进入 `val` 的数量，不是百分比。
- 所有样本先复制到 `train`，再按标签抽样复制到 `val`。
- `test` 目录会创建，但当前不会自动填充。
- `--uuid` 会把输出的图片和标签文件名改为 UUID。
- `--classes` 和 `--check` 参数在当前实现中尚未生效。
- 图片配对按 `Common.image_exts` 遍历同名扩展名（`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`）。

常用示例：

```shell
# 基本整理
yoloutils labelimg --source ./labelimg_data --target ./yolo_data

# 每个标签抽取 20 个样本到验证集
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --val 20

# 生成 UUID 文件名
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --uuid
```

#### 4.7.1 `auto`

`auto` 是独立子命令，用现有模型自动给图片打标，输出 YOLO 标签。

命令格式：

```shell
yoloutils auto \
    --source ./source \
    --target ./target \
    --model ./best.pt \
    --conf 0.5 \
    --csv ./report.csv \
    --output ./preview \
    --clean
```

帮助信息（当前源码）：

```shell
usage: yoloutils.py auto [-h] [--source SOURCE] [--target TARGET] [--clean] [--model best.pt] [--conf 0.5] [--csv report.csv] [--output /path/to/output]

options:
  -h, --help                show this help message and exit
  --source SOURCE           图片来源地址
  --target TARGET           图片目标地址
  --clean                   清理之前的数据
  --model best.pt           载入模型
  --conf 0.5                置信度阈值
  --csv report.csv          报告输出，哪些文件已经标准，哪些没有标注
  --output /path/to/output  输出标注效果
```

实现说明：

- 必填参数：`--source --target --model`。
- `--clean` 会先提示并确认，再删除 `target/output`。
- 按 `Common.image_exts` 递归扫描输入图片。
- 每张图片会复制到 `target` 对应相对路径，并生成同名 `.txt`：
  - 有检测框时写入 YOLO 标准标注；
  - 无检测框时生成空 `txt`。
- 会写 `target/classes.txt`（来源于模型 `names`）。
- 终端输出未标注文件表和统计汇总。
- `--csv` 导出“未标注文件名”列表（单列表头）。

常用示例：

```shell
# 基本自动打标
yoloutils auto --source ./source --target ./target --model ./best.pt

# 带置信度阈值和报表
yoloutils auto --source ./source --target ./target --model ./best.pt --conf 0.5 --csv ./report.csv
```

### 4.8 `resize`

按长边尺寸缩放图片，小于目标尺寸的图片会直接复制。

命令格式：

```shell
yoloutils resize \
    --source ./images \
    --target ./resized \
    --imgsz 640 \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py resize [-h] [--source SOURCE] [--target TARGET] [--clean]
                           [--imgsz 640] [--output ]

options:
  -h, --help       show this help message and exit
  --imgsz 640      长边尺寸
  --output         输出识别图像

通用参数:
  --source SOURCE  图片来源地址
  --target TARGET  图片目标地址
  --clean          清理之前的数据
```

详细使用说明（推荐流程）：

1. 选择目标尺寸：`--imgsz` 表示长边阈值，常见取值 `640/1280/1920`。
2. 指定输入输出目录，首次建议加 `--clean`，保证结果目录干净可复现。
3. 运行后关注统计表：`已处理` 表示发生缩放，`未处理` 表示直接复制。
4. 核验：随机检查几张大图与小图，确认大图已缩放、小图保持原样。

实现说明：

- 递归扫描 `source/**/*`，并按 `Common.image_exts` 过滤图片。
- 输出目录保留原始相对路径。
- 长边大于 `--imgsz` 时才会缩放，否则直接复制原图。
- 输出统计表中的“未处理”表示未缩放、直接复制的文件数。
- 当前源码声明了 `--output` 参数，但实际流程中没有使用该目录。

常用示例：

```shell
yoloutils resize --source ./images --target ./resized --imgsz 640
yoloutils resize --source ./images --target ./resized --imgsz 1920
```

### 4.9 `classify`

处理分类数据集，并自动生成 `train/test/val` 目录结构。支持在复制训练集之前先用检测模型裁剪图片。

命令格式：

```shell
yoloutils classify \
    --source ./source \
    --target ./dataset \
    --test 100 \
    --crop \
    --model ./best.pt \
    --output ./predict \
    --checklist ./checklist \
    --uuid \
    --verbose \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py classify [-h] [--source SOURCE] [--target TARGET]
                             [--clean] [--output ] [--checklist ] [--test 100]
                             [--crop] [--model ] [--uuid] [--verbose]

options:
  -h, --help       show this help message and exit
  --output         输出识别图像
  --checklist      输出识别图像
  --test 100       测试数量
  --crop           裁剪
  --model          裁剪模型
  --uuid           重命名图片为UUID
  --verbose        过程输出

通用参数:
  --source SOURCE  图片来源地址
  --target TARGET  图片目标地址
  --clean          清理之前的数据
```

输出目录结构：

```text
dataset/
├── train/
│   ├── class_a/
│   └── class_b/
├── test/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

详细使用说明（推荐流程）：

1. 准备分类目录：`source/<类别名>/*.jpg`，类别名就是最终目录名。
2. 先决定是否启用裁剪：
   - 纯分类数据整理：不加 `--crop`；
   - 先检测再入库：加 `--crop --model <pt>`。
3. 设置抽样数量：`--test` 会同时作用于 `test` 和 `val` 的每类抽样数量。
4. 核验：检查 `train/test/val` 每个类别子目录是否存在，并抽查图片是否符合预期（原图或裁剪图）。

实现说明：

- 首先把所有源图片复制或裁剪到 `train`。
- 然后从 `train` 中随机抽样复制到 `test` 和 `val`。
- `--test` 表示每个类别抽样多少张，同时用于 `test` 和 `val` 两个集合。
- `--crop` 生效时必须同时提供有效的 `--model`。
- 裁剪模式下，若一张图检测出多个框，会输出多个 `_0`、`_1` 之类的裁剪文件。
- `--output` 用于保存带检测框的推理结果。
- `--checklist` 用于保存多框样本的检查结果和裁剪产物，便于人工复核。
- `--uuid` 会将目标文件改名为 UUID。

常用示例：

```shell
# 不裁剪，直接划分分类数据集
yoloutils classify --source ./source --target ./dataset

# 每个类别抽样 50 张到 test 和 val
yoloutils classify --source ./source --target ./dataset --test 50

# 先检测裁剪，再做分类数据集
yoloutils classify \
    --source ./source \
    --target ./dataset \
    --crop \
    --model ./best.pt \
    --output ./predict \
    --checklist ./checklist
```

### 4.10 `test`

`test` 用于单模型批量推理目录中的图片，可选保存 CSV 和可视化图片。

命令格式：

```shell
yoloutils test \
    --source ./images \
    --model ./best.pt \
    --csv ./result.csv \
    --output ./predict
```

帮助信息（当前源码）：

```shell
usage: yoloutils.py test [-h] [--source SOURCE] [--target TARGET] [--clean]
                         [--model MODEL] [--csv result.csv] [--output OUTPUT]

options:
  -h, --help        show this help message and exit
  --source SOURCE   图片来源地址
  --target TARGET   图片目标地址
  --clean           清理之前的数据
  --model MODEL     模型路径
  --csv result.csv  保存结果
  --output OUTPUT   测试结果输出路径
```

详细使用说明（推荐流程）：

1. 准备输入目录：把要评估的图片放到 `--source`。
2. 传入 `--model` 指定单个模型。
3. 需要复盘时加 `--output`，需要归档时加 `--csv`。
4. 核验终端输出中的 `Total/Not found/Average`。

实现说明：

- 输入扫描：递归读取 `source/**/*`，并按 `Common.image_exts` 过滤（`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`，不区分大小写）。
- 必填 `--model`。
- 每张图片仅记录首个检测框的 `标签/置信度`。
- 表格列为：`文件, 标签, 置信度`。
- `--csv` 导出当前表格。
- `--output` 保存带框结果图。
- 该命令继承了通用参数中的 `--target`、`--clean`，当前仅使用 `--clean`（配合 `--output` 清理旧结果目录）。

#### 4.10.1 `diff` 子命令

`diff` 用于同一批图片上并发对比多个模型的检测表现。

命令格式：

```shell
yoloutils diff \
    --source ./images \
    -m ./best1.pt ./best2.pt ./best3.pt \
    -l person \
    -c ./diff.csv \
    -o ./predict_diff
```

帮助信息（当前源码）：

```shell
usage: yoloutils.py diff [-h] [--source SOURCE] [--target TARGET] [--clean]
                         [-m best1.pt best2.pt best3.pt [best1.pt best2.pt best3.pt ...]]
                         [-l ] [-o OUTPUT] [-c result.csv]

options:
  -h, --help            show this help message and exit
  --source SOURCE       图片来源地址
  --target TARGET       图片目标地址
  --clean               清理之前的数据
  -m, --model best1.pt best2.pt best3.pt [best1.pt best2.pt best3.pt ...]
                        模型
  -l, --label           标签过滤只统计指定标签
  -o, --output OUTPUT   对比结果输出路径
  -c, --csv result.csv  保存对比结果
```

关键参数：

- `-m/--model <m1> <m2> ...`：必填，对比模型列表。
- `-l/--label <name>`：可选，仅统计指定标签。
- `-c/--csv <file>`：可选，导出对比表。
- `-o/--output <dir>`：可选，导出可视化结果图（仅最后一个模型写图）。

执行与性能特性：

- 每个模型一个线程并发推理，不再按模型串行执行。
- 每个模型线程按文件维度推进进度，便于观察各模型实时速度差异。

结果说明：

- 对比表头：`文件, 标签, <模型1>, <模型2>, ...`。
- “标签”列是该图片在全部模型中的识别标签并集（去重后逗号拼接）。
- 每个模型列记录该模型在该图片上的最大置信度（保留两位小数）；未命中为 `0.00`。
- 终端汇总格式：`Total: N Average: modelA=x.xx, modelB=y.yy, ...`。

常用示例：

```shell
# 批量测试并保存 CSV
yoloutils test --source ./images --model ./best.pt --csv ./result.csv

# 同时保存可视化结果
yoloutils test --source ./images --model ./best.pt --output ./predict

# 对比三个模型，并导出对比 CSV
yoloutils diff --source ./images -m ./best1.pt ./best2.pt ./best3.pt -l person -c ./diff.csv

# 对比模型并导出可视化图
yoloutils diff --source ./images -m ./best1.pt ./best2.pt -o ./predict_diff
```

---

## 5. 通用参数

部分子命令复用以下参数：

| 参数 | 说明 |
|---|---|
| `--source` | 输入目录 |
| `--target` | 输出目录 |
| `--clean` | 运行前先删除目标目录中的旧数据 |

说明：

- `label` 和 `change` 不使用 `--target`。
- `test` 和 `diff` 当前不使用 `--target`；`--clean` 在提供 `--output` 时会先清理旧结果目录。

---

## 6. 使用建议

1. 批量操作前先备份原始数据，尤其是 `change` 和未指定 `--target` 的 `remove`。
2. 图片扩展名建议统一管理；当前支持 `.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`。
3. 使用 `--clean` 时确认目标目录没有需要保留的旧文件。
4. `labelimg`、`classify`、`test` 涉及随机抽样或推理，建议先在小目录上试跑一轮。
5. 若目录层级较深，优先保证文件名唯一，避免某些命令在输出时因只保留 basename 发生覆盖。

---

## 7. 常见问题

### Q1: 提示 `classes.txt 文件不存在`

确认以下几点：

- `source` 目录下存在 `classes.txt`
- 文件名大小写正确
- 运行目录和传入路径没有写错

### Q2: 标签统计结果和图片数对不上

`label --total` 统计的是标注框数量，不是图片数量。一张图中如果有多个框，会被累计多次。

### Q3: 为什么 `labelimg` 没有生成 `test` 数据

当前实现只会创建 `test` 目录，不会自动填充测试样本。

### Q4: 为什么某些图片没有被处理

优先检查：

- 图片是否为 `.jpg`
- 标注文件和图片是否同名
- 模型文件路径是否正确
- 输出目录是否被旧数据覆盖

---

## 8. 许可证

MIT License

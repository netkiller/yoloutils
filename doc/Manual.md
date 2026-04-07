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
| `crop` | 使用 YOLO 模型检测并裁剪图片 |
| `labelimg` | 将 labelimg 风格数据整理为 YOLO 训练目录 |
| `resize` | 按长边尺寸缩放图片 |
| `classify` | 处理分类数据集并划分 `train/test/val` |
| `test` | 用模型批量推理并输出表格或 CSV |

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
yoloutils classify -h
yoloutils test -h
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

使用 YOLO 模型批量检测图片，并把检测区域裁剪到目标目录。

命令格式：

```shell
yoloutils crop \
    --source ./images \
    --target ./cropped \
    --model ./best.pt \
    --output ./predict \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py crop [-h] [--source SOURCE] [--target TARGET] [--clean]
                         [--model best.pt] [--output /tmp/output]

options:
  -h, --help            show this help message and exit
  --model best.pt       模型
  --output /tmp/output  Yolo 输出目录

通用参数:
  --source SOURCE       图片来源地址
  --target TARGET       图片目标地址
  --clean               清理之前的数据
```

实现说明：

- 递归扫描 `source` 下的 `.jpg` 文件。
- `--target` 保留原始相对目录结构。
- `--model` 必填，使用 `ultralytics.YOLO` 加载模型。
- `--output` 存在时，会额外保存带检测框的推理结果，以及 `ultralytics` 生成的裁剪结果到 `output/crop/`。
- 当前实现每张图在 `target` 目录中只保留首个检测结果输出，目标文件名沿用原图相对路径。

常用示例：

```shell
yoloutils crop --source ./images --target ./cropped --model ./best.pt
```

### 4.7 `labelimg`

把 labelimg 风格的数据整理成 YOLO 训练目录结构，并生成 `data.yaml`。

命令格式：

```shell
yoloutils labelimg \
    --source ./labelimg_data \
    --target ./yolo_data \
    --val 10 \
    --uuid \
    --clean
```

帮助信息：

```shell
usage: yoloutils.py labelimg [-h] [--source SOURCE] [--target TARGET]
                             [--clean] [--classes CLASSES] [--val 10] [--uuid]
                             [--check]

options:
  -h, --help         show this help message and exit
  --classes CLASSES  classes.txt 文件
  --val 10           检验数量
  --uuid             输出文件名使用UUID
  --check            图片检查 corrupt JPEG restored and saved

通用参数:
  --source SOURCE    图片来源地址
  --target TARGET    图片目标地址
  --clean            清理之前的数据
```

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

- 程序实际读取的是 `source/classes.txt`。
- `--val` 表示每个标签随机抽取多少个样本进入 `val`，不是百分比。
- 所有样本会先进入 `train`，再从中抽样复制到 `val`。
- `test` 目录会创建，但当前实现不会自动填充测试集。
- `--uuid` 会把输出的图片和标签文件名改为 UUID。
- 当前源码中声明了 `--classes` 和 `--check` 参数，但这两个参数在实现里尚未真正生效。
- 图片配对逻辑主要按同名 `.jpg` 处理，准备数据时建议统一使用 `.jpg`。

常用示例：

```shell
# 基本整理
yoloutils labelimg --source ./labelimg_data --target ./yolo_data

# 每个标签抽取 20 个样本到验证集
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --val 20

# 生成 UUID 文件名
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --uuid
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

实现说明：

- 递归扫描 `source` 下的 `.jpg` 文件。
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

用 YOLO 模型批量推理目录中的图片，并输出表格结果，可选保存 CSV 和可视化图片。

命令格式：

```shell
yoloutils test \
    --source ./images \
    --model ./best.pt \
    --csv ./result.csv \
    --output ./predict
```

帮助信息：

```shell
usage: yoloutils.py test [-h] [--source SOURCE] [--target TARGET] [--clean]
                         [--model MODEL] [--csv result.csv] [--output OUTPUT]

options:
  -h, --help        show this help message and exit
  --model MODEL     模型路径
  --csv result.csv  保存测试结果
  --output OUTPUT   测试结果输出路径

通用参数:
  --source SOURCE   图片来源地址
  --target TARGET   图片目标地址
  --clean           清理之前的数据
```

实现说明：

- 会递归扫描 `source` 下除 `.txt` 和 `.DS_Store` 之外的文件。
- 每张图只记录首个检测结果的标签和置信度。
- 推理完成后会输出表格，并计算总数、未检出数量和平均置信度。
- `--csv` 会把结果保存为 CSV。
- `--output` 会保存带检测框的结果图片。
- 该命令虽然继承了通用参数中的 `--target`、`--clean`，但当前实现没有使用它们。

常用示例：

```shell
# 批量测试并保存 CSV
yoloutils test --source ./images --model ./best.pt --csv ./result.csv

# 同时保存可视化结果
yoloutils test --source ./images --model ./best.pt --output ./predict
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
- `test` 虽然带有 `--target`、`--clean` 帮助项，但当前实现未使用。

---

## 6. 使用建议

1. 批量操作前先备份原始数据，尤其是 `change` 和未指定 `--target` 的 `remove`。
2. 尽量统一使用 `.jpg` 图片，避免因为扩展名处理差异导致图片未被复制。
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

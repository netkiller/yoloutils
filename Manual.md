# yoloutils 操作手册

YOLO 标签工具集 - 用于处理 YOLO 目标检测数据集的标签和图片。

作者: Neo <netkiller@msn.com>
官网: https://www.netkiller.cn

---

## 目录

1. [安装指南](#安装指南)
2. [标签管理 (label)](#标签管理-label)
3. [合并标签 (merge)](#合并标签-merge)
4. [复制标签 (copy)](#复制标签-copy)
5. [删除标签 (remove)](#删除标签-remove)
6. [修改标签 (change)](#修改标签-change)
7. [图片裁剪 (crop)](#图片裁剪-crop)
8. [labelimg 转 YOLO (labelimg)](#labelimg-转-yolo-labelimg)
9. [修改图片尺寸 (resize)](#修改图片尺寸-resize)
10. [图像分类 (classify)](#图像分类-classify)

---

## 安装指南

### 方式一：使用 pip 安装

```shell
# 从 PyPI 安装
pip install netkiller-yoloutils

# 或从源码安装
pip install build
python -m build
pip install dist/netkiller_yoloutils-0.0.1-py3-none-any.whl --force-reinstall
```

### 方式二：开发模式安装

```shell
git clone https://github.com/netkiller/yoloutils.git
cd yoloutils
pip install -e .
```

### 方式三：手动安装

```shell
pip install setuptools wheel twine
python3 setup.py sdist
python3 setup.py install
```

---

## 标签管理 (label)

查看 classes.txt 文件、统计标签数量、搜索指定标签。

### 查看 classes.txt

```shell
yoloutils label --source /path/to/dataset --classes
```

### 统计标签图数量

```shell
yoloutils label --source /path/to/dataset --total
```

### 统计标签索引数量

```shell
yoloutils label --source /path/to/dataset --index
```

### 搜索指定标签

```shell
# 搜索索引为 1, 2, 3 的标签
yoloutils label --source /path/to/dataset --search 1 2 3
```

---

## 合并标签 (merge)

将两个目录中的 YOLO 标签 TXT 文件合并到新目录。

```shell
yoloutils merge \
    --left /path/to/dir1 \
    --right /path/to/dir2 \
    --output /path/to/output \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--left` | 左侧目录路径 |
| `--right` | 右侧目录路径 |
| `--output` | 最终输出目录 |
| `--clean` | 清理之前的数据 |

**注意：** 目标文件夹不能与原始图片文件夹相同。

---

## 复制标签 (copy)

根据指定标签从源目录复制图片和标签文件到目标目录。

```shell
yoloutils copy \
    --source /path/to/source \
    --target /path/to/target \
    --label person,dog \
    -u \
    -c
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址 |
| `--target` | 图片目标地址 |
| `--label` | 逗号分割的标签名称（如 `person,dog`） |
| `-u, --uuid` | 使用 UUID 作为输出文件名 |
| `-c, --clean` | 清理目标文件夹 |

**示例：**

```shell
# 复制包含 "cat" 标签的所有图片
yoloutils copy --source ./dataset --target ./output --label cat

# 复制多个标签
yoloutils copy --source ./dataset --target ./output --label person,car,bicycle

# 使用 UUID 重命名并清理目标
yoloutils copy --source ./dataset --target ./output --label dog -u -c
```

---

## 删除标签 (remove)

从 YOLO TXT 文件中删除指定标签。

```shell
yoloutils remove \
    --source /path/to/source \
    --target /path/to/target \
    --classes 1 2 3 \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址 |
| `--target` | 图片目标地址 |
| `--classes` | 标签序号（索引），可指定多个 |
| `--label` | 标签名称，可指定多个 |
| `--clean` | 清理之前的数据 |

**示例：**

```shell
# 按索引删除标签
yoloutils remove --source ./dataset --target ./output --classes 0 1 2

# 按名称删除标签
yoloutils remove --source ./dataset --target ./output --label cat dog

# 删除后清理目标目录
yoloutils remove --source ./dataset --target ./output --classes 0 --clean
```

---

## 修改标签 (change)

修改 YOLO TXT 文件中的标签索引。

```shell
yoloutils change \
    --source /path/to/dataset \
    --search 0 1 2 \
    --replace 3 4 5
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 目录路径 |
| `--search` | 要查找的标签序号 |
| `--replace` | 替换后的标签序号 |

**示例：**

```shell
# 将标签索引 0 替换为 5，1 替换为 6
yoloutils change --source ./dataset --search 0 1 --replace 5 6
```

---

## 图片裁剪 (crop)

使用 YOLO 模型检测并裁剪图片。

```shell
yoloutils crop \
    --source /path/to/source \
    --target /path/to/target \
    --model best.pt \
    --output /path/to/crops \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址 |
| `--target` | 图片目标地址 |
| `--model` | YOLO 模型文件路径 |
| `--output` | YOLO 输出目录（包含裁剪结果） |
| `--clean` | 清理之前的数据 |

**示例：**

```shell
# 使用模型裁剪图片
yoloutils crop --source ./images --target ./cropped --model best.pt
```

---

## labelimg 转 YOLO (labelimg)

将 labelimg 标注格式转换为 YOLO 训练数据集格式。

```shell
yoloutils labelimg \
    --source /path/to/labelimg \
    --target /path/to/yolo \
    --classes /path/to/classes.txt \
    --val 10 \
    --uuid \
    --check \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | labelimg 格式的标注文件目录 |
| `--target` | YOLO 格式输出目录 |
| `--classes` | classes.txt 文件路径 |
| `--val` | 验证集数量（默认 10） |
| `--uuid` | 使用 UUID 作为输出文件名 |
| `--check` | 检查图片完整性（损坏的 JPEG 会修复并保存） |
| `--clean` | 清理之前的数据 |

**输出目录结构：**

```
target/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── classes.txt
└── data.yaml
```

**示例：**

```shell
# 基本转换
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --classes ./classes.txt

# 转换为 YOLO 格式，10% 作为验证集，使用 UUID
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --classes ./classes.txt --val 10 --uuid

# 清理后重新转换
yoloutils labelimg --source ./labelimg_data --target ./yolo_data --classes ./classes.txt --clean
```

---

## 修改图片尺寸 (resize)

修改图片尺寸，保持长边为指定大小。

```shell
yoloutils resize \
    --source /path/to/source \
    --target /path/to/target \
    --imgsz 640 \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址 |
| `--target` | 图片目标地址 |
| `--imgsz` | 长边尺寸（默认 640） |
| `--output` | 输出识别图像目录 |
| `--clean` | 清理之前的数据 |

**示例：**

```shell
# 将图片长边调整为 640 像素
yoloutils resize --source ./images --target ./resized --imgsz 640

# 调整为 1920 像素
yoloutils resize --source ./images --target ./resized --imgsz 1920
```

---

## 图像分类 (classify)

处理分类数据集，自动划分为 train/test/val 目录结构。

```shell
yoloutils classify \
    --source /path/to/source \
    --target /path/to/target \
    --test 100 \
    --crop \
    --model best.pt \
    --uuid \
    --verbose \
    --clean
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址（目录结构：source/classname/image.jpg） |
| `--target` | 图片目标地址 |
| `--test` | 每个类别测试集数量（默认 10） |
| `--crop` | 启用裁剪功能 |
| `--model` | 裁剪使用的 YOLO 模型 |
| `--uuid` | 使用 UUID 作为输出文件名 |
| `--verbose` | 输出详细过程信息 |
| `--output` | 输出识别图像目录 |
| `--checklist` | 输出检查列表目录 |
| `--clean` | 清理之前的数据 |

**输入目录结构：**

```
source/
├── cat/
│   ├── image1.jpg
│   └── image2.jpg
├── dog/
│   ├── image3.jpg
│   └── image4.jpg
```

**输出目录结构：**

```
target/
├── train/
│   ├── cat/
│   └── dog/
├── test/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

**示例：**

```shell
# 基本分类处理
yoloutils classify --source ./images --target ./dataset

# 每个类别取 100 张作为测试集
yoloutils classify --source ./images --target ./dataset --test 100

# 使用模型裁剪并生成检查列表
yoloutils classify --source ./images --target ./dataset --crop --model best.pt --checklist ./checklist

# 使用 UUID 重命名
yoloutils classify --source ./images --target ./dataset --uuid
```

---

## 通用参数

以下参数适用于多个子命令：

| 参数 | 说明 |
|------|------|
| `--source` | 图片来源地址 |
| `--target` | 图片目标地址 |
| `--clean` | 清理之前的数据（先删除目标目录） |

---

## 注意事项

1. **YOLO TXT 格式**：工具处理的标签文件格式为 `index x_center y_center width height`，其中 index 从 0 开始。

2. **classes.txt**：某些操作需要 classes.txt 文件，格式为每行一个类别名称。

3. **路径处理**：所有路径建议使用绝对路径，避免相对路径带来的问题。

4. **数据备份**：使用 `--clean` 参数前请确保已备份重要数据，该操作会删除目标目录。

5. **依赖项**：确保安装了所有依赖，包括 `ultralytics`（YOLO 库）、`opencv-python`、`pillow`、`pyyaml`、`tqdm`、`texttable`。

---

## 常见问题

**Q: 显示 "classes.txt 文件不存在"**
A: 确保源目录包含 classes.txt 文件，且路径正确。

**Q: 标签索引超出范围**
A: 检查 classes.txt 中的类别数量，确保标签索引在有效范围内（0 到 len(classes)-1）。

**Q: 图片找不到**
A: 确保 TXT 文件对应的图片文件（JPG/PNG）存在于相同目录。

---

## 许可证

MIT License

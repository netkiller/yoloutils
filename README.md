# yoloutils - YOLO Utilities

YOLO 标签工具集 - 用于处理 YOLO 目标检测数据集的标签和图片。

作者: Neo <netkiller@msn.com>
官网: https://www.netkiller.cn

## 从 PyPI 安装

使用 pip 安装

```shell
pip install netkiller-yoloutils
```

## 帮助信息

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils
usage: yoloutils.py [-h] {label,merge,copy,remove,change,crop,labelimg,auto,resize,classify,test,diff,image,workstation} ...

Yolo 标签与图像处理工具

options:
  -h, --help            show this help message and exit

子命令:
  工具含标签类处理和图像类处理工具

  {label,merge,copy,remove,change,crop,labelimg,auto,resize,classify,test,diff,image,workstation}
                        风险提示：当使用 --clean 参数时会删除目标目录和输出目录
    label               标签统计、索引统计、标签搜索
    merge               合并两个TXT文件中的标签到新TXT文件
    copy                从指定标签复制图片文件
    remove              从YOLO TXT文件中删除指定标签
    change              修改标签索引
    crop                图片裁剪
    labelimg            labelimg 格式转换为 yolo 训练数据集
    auto                用现有模型自动给训练图像打标签
    resize              修改图片尺寸
    classify            图像分类数据处理
    test                模型测试工具
    diff                模型比较工具
    image               图像工具
    workstation         Yolo 工作站

Author: netkiller - https://www.netkiller.cn
```

## 标签管理

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils label -h
usage: yoloutils label [-h] [--source /tmp/dir1] [--classes] [--total] [--index] [--search 1 2 3 [1 2 3 ...]]

options:
  -h, --help            show this help message and exit
  --source /tmp/dir1    目录
  --classes             查看 classes.txt 文件
  --total               统计标签图数量
  --index               统计标签索引数量
  --search 1 2 3 [1 2 3 ...]
                        搜索标签
```

## 合并标签

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils merge -h
usage: yoloutils merge [-h] [--left /tmp/dir1] [--right /tmp/dir2] [--output /tmp/output] [--clean]

options:
  -h, --help            show this help message and exit
  --left /tmp/dir1      左侧目录
  --right /tmp/dir2     右侧目录
  --output /tmp/output  最终输出目录
  --clean               清理之前的数据


```

## 复制标签

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils copy -h
usage: yoloutils copy [-h] [--source SOURCE] [--target TARGET] [--label LABEL] [-u] [-c]

options:
  -h, --help       show this help message and exit
  --source SOURCE  图片来源地址
  --target TARGET  图片目标地址
  --label LABEL    逗号分割多个标签
  -u, --uuid       UUID 文件名
  -c, --clean      清理目标文件夹


```

## 删除标签

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils remove -h
usage: yoloutils remove [-h] [--source SOURCE] [--target TARGET] [--clean] [--classes 1 2 3 [1 2 3 ...]]
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

## 修改标签

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils change -h
usage: yoloutils change [-h] [--source /tmp/dir1] [--search 1 2 3 [1 2 3 ...]] [--replace 4 5 6 [4 5 6 ...]]

options:
  -h, --help            show this help message and exit
  --source /tmp/dir1    目录
  --search 1 2 3 [1 2 3 ...]
                        标签序号
  --replace 4 5 6 [4 5 6 ...]
                        标签名称


```

## 裁剪图片

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils crop -h  
usage: yoloutils crop [-h] [--source SOURCE] [--target TARGET] [--clean] [--model best.pt] [--output /tmp/output]

options:
  -h, --help            show this help message and exit
  --model best.pt       模型
  --output /tmp/output  Yolo 输出目录

通用参数:
  --source SOURCE       图片来源地址
  --target TARGET       图片目标地址
  --clean               清理之前的数据

```

## labelimg 转 yolo 训练数据集

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils labelimg -h
usage: yoloutils labelimg [-h] [--source SOURCE] [--target TARGET] [--clean] [--classes CLASSES] [--val 10] [--uuid] [--check]

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

## 自动打标签

```shell
(.venv) neo@netkiller yoloutils % yoloutils auto
usage: yoloutils auto [-h] [--source SOURCE] [--target TARGET] [--clean] [--model best.pt] [--conf 0.5] [--csv report.csv] [--output /path/to/output]

options:
  -h, --help                show this help message and exit
  --source SOURCE           图片来源地址
  --target TARGET           图片目标地址
  --clean                   清理之前的数据
  --model best.pt           载入模型
  --conf 0.5                置信度阈值
  --csv report.csv          报告输出，哪些文件已经标准，哪些没有标注
  --output /path/to/output  输出标注效果

用载入的模型自动给目录中的文件打标
```

## 修改图片尺寸

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils resize -h  
usage: yoloutils resize [-h] [--source SOURCE] [--target TARGET] [--clean] [--imgsz 640] [--output ]

options:
  -h, --help       show this help message and exit
  --imgsz 640      长边尺寸
  --output         输出识别图像

通用参数:
  --source SOURCE  图片来源地址
  --target TARGET  图片目标地址
  --clean          清理之前的数据
```

## 图像查找

```shell
# 查找长边大于 1920 的图片
yoloutils image --source ./images --imgsz '>1920' --csv ./result.csv

# 查找长边小于 1920 的图片
yoloutils image --source ./images --imgsz '<1920'
```

`>` 和 `<` 在 shell 中是重定向符，使用时需要加引号或转义。
`--csv` 可将查询结果导出为 CSV，列为：`文件, 宽, 高`。

```shell
# 检查异常 JPG/JPEG 图片
yoloutils image --source ./images --check --csv ./jpg-check.csv
```

`--check` 递归检查 `.jpg/.jpeg` 文件，发现异常时输出并可导出 CSV，列为：`文件, 错误`。

## 工作站

```shell
python src/netkiller/yoloutils/yoloutils.py workstation -w /Users/neo/tmp/yolo/source
```

启动本地 FastAPI 站点，默认地址为 `http://127.0.0.1:8000`。页面包含目录树、文件列表、图像预览和右侧信息栏；文件列表中有效标注显示绿色，空 `.txt`、无效 `.txt` 或损坏图片显示红色；图像存在同名 `.txt` 时会按 YOLO 标注绘制 box 框。图像栏头部提供自动标注激活、删除、重置和保存按钮。目录树和文件列表、文件列表和图像区域都可左右拖动调整比例，目录栏可隐藏，隐藏后图像栏填充释放空间。右侧信息栏可隐藏；标签下方展示当前图片 RGB 直方图，标签和直方图可上下拖动分配比例，直方图可折叠；标签/直方图区域和 EXIF 可上下拖动调整比例，EXIF 可折叠到底部。底部 footer 左侧展示 `--workspace` 位置，右侧展示图像数量、`.txt` 数量、损坏图像和无效 `.txt` 数量。

后台运行：

```shell
python src/netkiller/yoloutils/yoloutils.py workstation -w /Users/neo/tmp/yolo/source -d
```

后台模式会在工作目录写入 `.yoloutils-workstation.pid` 和 `.yoloutils-workstation.log`。

## 图像分类数据处理

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils classify -h
usage: yoloutils classify [-h] [--source SOURCE] [--target TARGET] [--clean] [--output ] [--checklist ] [--test 100] [--crop] [--model ]
                          [--uuid] [--verbose]

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

## 模型测试

```shell
usage: yoloutils test [-h] [--source SOURCE] [--target TARGET] [--clean] [--model MODEL] [--csv result.csv] [--output OUTPUT]

options:
  -h, --help        show this help message and exit
  --source SOURCE   图片来源地址
  --target TARGET   图片目标地址
  --clean           清理之前的数据
  --model MODEL     模型路径
  --csv result.csv  保存结果
  --output OUTPUT   测试结果输出路径
```

## 模型对比

```shell
(.venv) neo@netkiller yoloutils % yoloutils diff 
usage: yoloutils diff [-h] [--source SOURCE] [--target TARGET] [--clean] [-m best1.pt best2.pt best3.pt [best1.pt best2.pt best3.pt ...]] [-l ] [-o OUTPUT] [-c result.csv]

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

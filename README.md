# yoloutils
YOLO Utilities

## 帮助信息

```shell
(.venv) neo@Neo-Mac-mini-M4 yoloutils % yoloutils
usage: yoloutils [-h] {label,merge,copy,remove,change,crop,labelimg,resize,classify} ...

Yolo 标签工具

options:
  -h, --help            show this help message and exit

subcommands:
  valid subcommands

  {label,merge,copy,remove,change,crop,labelimg,resize,classify}
                        additional help
    label               标签处理工具
    merge               合并两个TXT文件中的标签到新TXT文件
    copy                从指定标签复制图片文件
    remove              从YOLO TXT文件中删除指定标签
    change              修改标签索引
    crop                图片裁剪
    labelimg            labelimg 格式转换为 yolo 训练数据集
    resize              修改图片尺寸
    classify            图像分类数据处理

Author: netkiller - https://www.netkiller.cn
```

### 标签管理

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

### 合并标签

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

### 复制标签

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

### 删除标签

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

### 修改标签 

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

### 裁剪图片

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

### labelimg 转 yolo 训练数据集

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

### 修改图片尺寸

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

### 图像分类数据处理

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
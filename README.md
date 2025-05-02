# yoloutils
YOLO Utilities

## requirements

```shell
pip install --upgrade pip
# pip install -r requirements.txt
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# PS D:\workspace\medical> .\.venv\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Install

### setuptools

```shell
    $ pip install setuptools wheel twine
	$ cd /usr/local/src/
	$ git clone https://github.com/netkiller/yoloutils.git
	$ cd yoloutils
	$ python3 setup.py sdist
	$ python3 setup.py install --prefix=/srv/yoloutils
```

### RPM 包

```shell
    $ python setup.py bdist_rpm

```

### Windows 文件

```shell
    $ python setup.py bdist_wininst
```

## Deploy Pypi

```shell

	$ pip install setuptools wheel twine
	$ python setup.py sdist bdist_wheel
	$ twine upload dist/netkiller-devops-x.x.x.tar.gz 

```

## pyproject.toml

```shell
pip install build
python -m build
pip install dist/yoloutils-0.0.1-py3-none-any.whl --force-reinstall

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
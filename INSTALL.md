# Install

## requirements

```shell
pip install --upgrade pip
# pip install -r requirements.txt
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# PS D:\workspace\medical> .\.venv\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## setuptools

```shell
    $ pip install setuptools wheel twine
	$ cd /usr/local/src/
	$ git clone https://github.com/netkiller/yoloutils.git
	$ cd yoloutils
	$ python3 setup.py sdist
	$ python3 setup.py install --prefix=/srv/yoloutils
```

## RPM 包

```shell
    $ python setup.py bdist_rpm

```

## Windows 文件

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

(.venv) neo@Neo-Mac-mini-M4 yoloutils % twine upload dist/netkiller_yoloutils-0.0.1*

(.venv) neo@Neo-Mac-mini-M4 yoloutils % pip install netkiller-yoloutils
```
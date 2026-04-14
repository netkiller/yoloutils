import glob
import logging
import os
import shutil
import sys
import uuid

from texttable import Texttable
from tqdm import tqdm

try:
    from . import BASE_DIR, Common
except ImportError:
    # Support direct script execution (python label.py ...)
    from __init__ import BASE_DIR, Common


class YoloLabelRemove(Common):
    total = {"change": 0, "remove": 0, "skip": 0, "error": 0}

    def __init__(self, parser, args):
        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        parser.add_argument(
            "--classes", nargs="+", default=None, help="标签序号", metavar="1 2 3"
        )
        parser.add_argument(
            "--label", nargs="+", default=None, help="标签名称", metavar="label1 label2"
        )
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("remove")
        self.indexs = []

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)
        return files

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.target):
                    shutil.rmtree(self.args.target)
            if self.args.target:
                os.makedirs(self.args.target, exist_ok=True)

            self.files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)

            if self.args.label:
                classes = os.path.join(self.args.source, "classes.txt")
                if not os.path.isfile(classes):
                    print(f"classes.txt 文件不存在: {classes}")
                    self.logger.error("classes.txt 文件不存在！")
                    exit()
                else:
                    with open(classes) as file:
                        n = 0
                        for line in file.readlines():
                            if line.strip() in self.args.label:
                                self.indexs.append(n)
                            n += 1
            if self.args.classes:
                for index in self.args.classes:
                    self.indexs.append(int(index))
            self.logger.info(f"remove classes len={len(self.indexs)} indexs={self.indexs}")
        except Exception as e:
            self.logger.error("input: ", repr(e))
            exit()

    def process(self):
        with tqdm(total=len(self.files), ncols=150) as progress:
            for file in self.files:
                progress.set_description(file)
                filename = os.path.basename(file)
                try:
                    if filename.lower() == "classes.txt":
                        progress.update(1)
                        self.total["skip"] += 1
                        self.logger.info(f"skip file={file}")
                        continue
                    else:
                        if self.args.target:
                            target = os.path.join(self.args.target, filename)
                        else:
                            target = file
                        lines = []
                        isChange = False
                        with open(file, "r") as original:
                            for line in original.readlines():
                                index = int(line.strip().split(" ")[0])
                                if index in self.indexs:
                                    self.logger.info(f"index={index} indexs={self.indexs}")
                                    isChange = True
                                    continue
                                lines.append(line)
                        if len(lines) > 0:
                            if isChange:
                                with open(target, "w") as newfile:
                                    newfile.writelines(lines)
                                self.total["change"] += 1
                                self.logger.info(f"change target={target}")
                        else:
                            os.remove(target)
                            os.remove(target.replace(".txt", ".jpg"))
                            self.total["remove"] += 1
                            self.logger.info(f"remove target={target}")

                except FileNotFoundError as e:
                    self.logger.error(str(e))
                    self.total["error"] += 1

                progress.update(1)

    def output(self):
        tables = [["操作", "处理"]]
        tables.append(["count", len(self.files)])
        for k, v in self.total.items():
            tables.append([k, v])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self):
        if self.args.source and (self.args.classes or self.args.label):
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()


class YoloLabelMerge(Common):
    lose = []

    def __init__(self, parser, args):
        parser.add_argument(
            "--left", type=str, default=None, help="左侧目录", metavar="/tmp/dir1"
        )
        parser.add_argument(
            "--right", default=None, type=str, help="右侧目录", metavar="/tmp/dir2"
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="最终输出目录",
            metavar="/tmp/output",
        )
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        self.parser = parser
        self.args = args

        self.basedir = BASE_DIR
        sys.path.append(self.basedir)

    def scanfile(self, path):
        files = glob.glob(path)
        return files

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)
        return files

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.output):
                    shutil.rmtree(self.args.output)
            os.makedirs(self.args.output, exist_ok=True)

            self.lefts = self.scanfile(os.path.join(self.args.left, "*.txt"))
            self.rights = self.scanfile(os.path.join(self.args.right, "*.txt"))
        except Exception as e:
            self.logger.error(e)
            print("input: ", e)
            exit()

    def process(self):
        with tqdm(total=len(self.lefts), ncols=100) as progress:
            for file in self.lefts:
                progress.set_description(file)
                filename = os.path.basename(file)
                try:
                    if filename.lower() == "classes.txt":
                        shutil.copyfile(file, os.path.join(self.args.output, filename))
                    else:
                        left = os.path.join(self.args.left, filename)
                        right = os.path.join(self.args.right, filename.replace("_0.", "."))
                        output = os.path.join(self.args.output, filename)
                        image = filename.replace(".txt", ".jpg")

                        shutil.copyfile(
                            os.path.join(self.args.left, image),
                            os.path.join(self.args.output, image),
                        )

                        if not os.path.isfile(right):
                            shutil.copyfile(left, output)
                        else:
                            with (
                                open(left, "r") as file1,
                                open(right, "r") as file2,
                                open(output, "w") as file_out,
                            ):
                                txt1 = file1.read()
                                txt2 = file2.read()

                                file_out.write(txt1)
                                file_out.write(txt2)

                except FileNotFoundError as e:
                    print(str(e))
                    self.lose.append(e.filename)
                    exit()

                progress.update(1)

    def output(self):
        if not self.lose:
            return
        tables = [["丢失文件"]]
        for file in self.lose:
            tables.append([file])
        tables.append([f"合计：{len(self.lose)}"])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self):
        if self.args.left and self.args.right:
            if self.args.left == self.args.right:
                print("目标文件夹不能与原始图片文件夹相同")
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()


class YoloLabelCopy(Common):
    def __init__(self, parser, args):
        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--label", type=str, default=None, help="逗号分割多个标签"
        )
        parser.add_argument(
            "-u", "--uuid", action="store_true", default=False, help="UUID 文件名"
        )
        parser.add_argument(
            "-c", "--clean", action="store_true", default=False, help="清理目标文件夹"
        )
        self.parser = parser
        self.args = args

        self.classes = {}
        self.lables = []
        self.missed = []
        self.count = 0

        self.logger = logging.getLogger("copy")

    def input(self):
        if self.args.clean:
            if os.path.exists(self.args.target):
                shutil.rmtree(self.args.target)

        os.makedirs(os.path.join(self.args.target), exist_ok=True)

        classes = os.path.join(self.args.source, "classes.txt")
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error("classes.txt 文件不存在！")
            exit()
        else:
            tables = [["序号", "标签"]]
            with open(classes) as file:
                n = 0
                for line in file:
                    self.classes[line.strip()] = n
                    tables.append([n, line.strip()])
                    n += 1
                self.logger.info(f"classes len={len(self.classes)} dict={self.classes}")
            if self.args.label:
                for label in self.args.label.split(","):
                    if label in self.classes.keys():
                        self.lables.append(self.classes[label])
                    else:
                        self.logger.error(f"label {label} 不存在")
                        exit()
                self.logger.info(f"label len={len(self.lables)} list={self.lables}")

        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

        self.files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)

    def process(self):
        with (
            tqdm(total=len(self.files), ncols=150) as processBar,
            tqdm(total=len(self.files), ncols=150) as processBarImage,
        ):
            for file in self.files:
                processBar.set_description(f"{file}")

                if file.endswith("classes.txt"):
                    processBar.update(1)
                    processBarImage.update(1)
                    self.logger.info("skip classes.txt")
                    continue

                source = file
                if self.args.uuid:
                    uuid4 = uuid.uuid4()
                    target = os.path.join(self.args.target, f"{uuid4}.txt")
                else:
                    target = os.path.join(self.args.target, os.path.basename(source))

                image = file.replace(".txt", ".jpg", 1)
                processBarImage.set_description(f"{image}")

                if self.args.label:
                    with open(file) as txt:
                        for line in txt.readlines():
                            index = int(line.strip().split(" ")[0])
                            if index in self.lables:
                                shutil.copy(source, target)
                                self.logger.info(f"copy source={source} target={target}")
                                source = source.replace(".txt", ".jpg")
                                target = self.args.target.replace(".txt", ".jpg")
                                shutil.copy(source, target)
                                self.logger.info(f"copy source={source} target={target}")
                                self.count += 1
                                break
                else:
                    shutil.copy(source, target)
                    self.logger.info(f"copy source={file} target={target}")
                try:
                    shutil.copy(image, target.replace(".txt", ".jpg", 1))
                except FileNotFoundError as e:
                    self.logger.error(e)
                processBar.update(1)
                processBarImage.update(1)

    def output(self):
        shutil.copy(f"{self.args.source}/classes.txt", f"{self.args.target}/classes.txt")

        tables = [["输出", "处理"]]
        tables.append([len(self.files), self.count])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self):
        if self.args.source and self.args.target:
            self.logger.info("Start")
            self.input()
            self.process()
            self.output()
            self.logger.info("Done")
        else:
            self.parser.print_help()
            exit()


class YoloLabelChange(Common):
    count = 0

    def __init__(self, parser, args):
        parser.add_argument(
            '-s',
            "--source", type=str, default=None, help="目录", metavar="/tmp/dir1"
        )
        parser.add_argument(
            '-f',
            "--find", nargs="+", default=None, help="标签序号", metavar="1 2 3"
        )
        parser.add_argument(
            '-r',
            "--replace", nargs="+", default=None, help="标签名称", metavar="4 5 6"
        )
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.editable = {}
        self.total = {}

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)
        return files

    def input(self):
        try:
            self.logger.info(f"search={self.args.find}")
            self.logger.info(f"replace={self.args.replace}")

            for n in range(0, len(self.args.find)):
                self.editable[self.args.find[n]] = self.args.replace[n]

            self.logger.info(f"editable={self.editable}")

            self.files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)
            self.logger.info(f"files total={len(self.files)}")
        except Exception as e:
            self.logger.error("input: ", e)
            exit()

    def process(self):
        with tqdm(total=len(self.files), ncols=150) as progress:
            for file in self.files:
                progress.set_description(file)
                filename = os.path.basename(file)
                self.logger.info(f"file={file}")
                try:
                    if filename.lower() == "classes.txt":
                        progress.update(1)
                        self.logger.info(f"skip file={file}")
                        continue
                    else:
                        lines = []
                        with open(file, "r", encoding="utf-8") as original:
                            for line in original.readlines():
                                if not line.strip():
                                    self.logger.info(f"null line={line}")
                                    continue
                                index = int(line.strip().split(" ")[0])

                                for s, r in self.editable.items():
                                    if line.startswith(f"{s} "):
                                        line = line.replace(f"{s} ", f"{r} ", 1)
                                        self.logger.info(
                                            f"search={s} replace={r} line={line.strip()}"
                                        )
                                        break

                                if index not in self.total:
                                    self.total[index] = 0
                                self.total[index] += 1

                                lines.append(line)
                        if len(lines) > 0:
                            with open(file, "w", encoding="utf-8") as newfile:
                                newfile.writelines(lines)
                                self.logger.info(f"save file={file} text={lines}")

                except FileNotFoundError as e:
                    print(str(e))
                    exit()

                progress.update(1)

    def output(self):
        if len(self.total) == 0:
            return
        tables = [["索引", "数量"]]
        for k, v in self.total.items():
            tables.append([k, v])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self):
        if self.args.source and self.args.find and self.args.replace:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()


class YoloLabel(Common):
    count = 0

    def __init__(self, parser, args):
        parser.add_argument('-s',
                            "--source", type=str, default=None, help="目录", metavar="/tmp/dir1"
                            )
        parser.add_argument(
            '-c',
            "--classes",
            action="store_true",
            default=False,
            help="查看 classes.txt 文件",
        )
        parser.add_argument(
            '-t',
            "--total", action="store_true", default=False, help="统计标签图数量"
        )
        parser.add_argument(
            '-i',
            "--index", action="store_true", default=False, help="统计标签索引数量"
        )
        parser.add_argument(
            '-f',
            "--find", nargs="+", default=None, help="搜索标签", metavar="1 2 3"
        )
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.indexs = {}

    def classes(self):
        classes = os.path.join(self.args.source, "classes.txt")
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error("classes.txt 文件不存在！")
            exit()
        else:
            tables = [["序号", "标签"]]
            with open(classes) as file:
                n = 0
                for line in file:
                    tables.append([n, line.strip()])
                    n += 1

            table = Texttable(max_width=100)
            table.add_rows(tables)
            print(table.draw())

    def total(self):
        self.files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)
        self.logger.info(f"files total={len(self.files)}")
        with tqdm(total=len(self.files), ncols=150) as progress:
            for file in self.files:
                progress.set_description(file)
                filename = os.path.basename(file)
                self.logger.info(f"file={file}")
                try:
                    if filename.lower() == "classes.txt":
                        progress.update(1)
                        self.logger.info(f"skip file={file}")
                        continue
                    else:
                        with open(file, "r", encoding="utf-8") as original:
                            for line in original.readlines():
                                if line.strip():
                                    index = int(line.strip().split(" ")[0])
                                    if index in self.indexs.keys():
                                        self.indexs[index] += 1
                                    else:
                                        self.indexs[index] = 1

                except FileNotFoundError as e:
                    print(str(e))
                    exit()

                progress.update(1)

        if len(self.indexs) == 0:
            return

        if self.args.index:
            tables = [["索引", "数量"]]
            for k, v in self.indexs.items():
                tables.append([k, v])
        else:
            classes = os.path.join(self.args.source, "classes.txt")
            if not os.path.isfile(classes):
                print(f"classes.txt 文件不存在: {classes}")
                self.logger.error("classes.txt 文件不存在！")
                exit()
            else:
                with open(classes) as file:
                    labels = file.readlines()
                    tables = [["标签", "索引", "数量"]]
                    for k, v in self.indexs.items():
                        try:
                            tables.append([labels[k], k, v])
                        except IndexError as e:
                            tables.append(["", k, v])
                            self.logger.error(e)
        self.logger.info(f"tables len={len(tables)} data={tables}")
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def search(self):
        self.files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)
        self.logger.info(f"files total={len(self.files)}")
        data = {}

        with tqdm(total=len(self.files), ncols=100) as progress:
            for file in self.files:
                filename = os.path.basename(file)
                self.logger.info(f"file={file}")
                try:
                    if filename.lower() == "classes.txt":
                        self.logger.info(f"skip file={file}")
                        continue
                    else:
                        with open(file, "r", encoding="utf-8") as original:
                            for line in original.readlines():
                                index = line.strip().split(" ")[0]
                                if index not in data.keys():
                                    data[index] = []
                                if index in self.args.find:
                                    data[index].append(file)

                except FileNotFoundError as e:
                    print(str(e))
                    exit()

                progress.update(1)

        if len(data) == 0:
            return
        tables = [["索引", "文件"]]
        for k, v in data.items():
            if v:
                tables.append([k, v])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self):
        if self.args.classes and self.args.source:
            self.classes()
        elif self.args.source and self.args.total:
            self.total()
        elif self.args.source and self.args.index:
            self.total()
        elif self.args.source and self.args.find:
            self.search()
        else:
            self.parser.print_help()
            exit()

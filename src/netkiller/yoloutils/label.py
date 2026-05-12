import csv
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
    def __init__(self):

        self.logger = logging.getLogger("remove")
        self.indexs = set()
        self.changes = []
        self.total = {"change": 0, "classes.txt": 0, "error": 0}

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)
        return files

    def input(self):
        try:
            if not os.path.isdir(self.args.source):
                print(f"source 目录不存在: {self.args.source}")
                self.logger.error(f"source 目录不存在: {self.args.source}")
                exit()

            self.files = sorted(glob.glob(f"{self.args.source}/**/*.txt", recursive=True))

            if self.args.classes:
                classes = os.path.join(self.args.source, "classes.txt")
                if not os.path.isfile(classes):
                    print(f"classes.txt 文件不存在: {classes}")
                    self.logger.error("classes.txt 文件不存在！")
                    exit()
                else:
                    with open(classes) as file:
                        labels = {
                            line.strip(): index
                            for index, line in enumerate(file)
                            if line.strip()
                        }
                    for label in self.args.classes:
                        if label in labels:
                            self.indexs.add(labels[label])
                        else:
                            print(f"classes.txt 中未找到标签: {label}")
                            self.logger.warning(f"label not found: {label}")
            if self.args.index:
                for index in self.args.index:
                    self.indexs.add(int(index))
            if not self.indexs:
                print("没有可删除的标签索引")
                self.logger.error("empty remove indexs")
                exit()
            self.logger.info(f"remove classes len={len(self.indexs)} indexs={sorted(self.indexs)}")
        except Exception as e:
            self.logger.error(f"input: {repr(e)}")
            exit()

    def add_change(self, action, file, removed_lines):
        self.changes.append(
            {
                "action": action,
                "file": os.path.relpath(file, self.args.source),
                "lines": removed_lines,
            }
        )

    def process(self):
        with tqdm(total=len(self.files), ncols=120) as progress:
            for file in self.files:
                progress.set_description(os.path.relpath(file, self.args.source))
                filename = os.path.basename(file)
                try:
                    if filename.lower() == "classes.txt":
                        progress.update(1)
                        self.total["classes.txt"] += 1
                        self.logger.info(f"skip file={file}")
                        continue
                    else:

                        lines = []
                        removed_lines = []
                        with open(file, "r") as original:
                            for line_number, line in enumerate(original.readlines(), start=1):
                                stripped = line.strip()
                                if not stripped:
                                    lines.append(line)
                                    continue
                                try:
                                    index = int(stripped.split()[0])
                                except (ValueError, IndexError):
                                    lines.append(line)
                                    self.total["error"] += 1
                                    self.logger.error(f"invalid label line file={file} line={line_number} text={stripped}")
                                    continue
                                if index in self.indexs:
                                    self.logger.info(f"index={index} indexs={sorted(self.indexs)}")
                                    removed_lines.append(f"{line_number}:{stripped}")
                                    continue
                                lines.append(line)

                        if not removed_lines:
                            progress.update(1)
                            continue

                        self.add_change("change", file, removed_lines)
                        if not self.args.dry_run:
                            with open(file, "w") as newfile:
                                newfile.writelines(lines)
                        self.total["change"] += 1
                        self.logger.info(f"change target={file}")

                except FileNotFoundError as e:
                    self.logger.error(str(e))
                    self.total["error"] += 1

                progress.update(1)

    def output(self):
        if self.args.csv:
            report_dir = os.path.dirname(self.args.csv)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(self.args.csv, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("操作", "TXT 文件", "删除行"))
                for change in self.changes:
                    writer.writerow(
                        (
                            change["action"],
                            change["file"],
                            "; ".join(change["lines"]),
                        )
                    )

        if self.args.dry_run:
            print("DRY-RUN: 以下 .txt 文件将被修改，实际文件未变更。")
            tables = [["操作", "TXT 文件", "删除行"]]
            for change in self.changes:
                tables.append(
                    [
                        change["action"],
                        change["file"],
                        "\n".join(change["lines"]),
                    ]
                )
            if len(tables) == 1:
                tables.append(["无", "", ""])
            table = Texttable(max_width=160)
            table.add_rows(tables)
            print(table.draw())

        tables = [["操作", "处理"]]
        tables.append(["count", len(self.files)])
        for k, v in self.total.items():
            tables.append([k, v])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

    def main(self, args):
        self.args = args
        if self.args.source and (self.args.classes or self.args.index):
            self.input()
            self.process()
            self.output()


class YoloLabelMerge(Common):
    lose = []

    def __init__(self, parser, args):
        parser.add_argument(
            '-l',
            "--left", type=str, default=None, help="左侧目录", metavar="/tmp/dir1"
        )
        parser.add_argument(
            '-r',
            "--right", default=None, type=str, help="右侧目录", metavar="/tmp/dir2"
        )
        parser.add_argument(
            '-o',
            "--output",
            type=str,
            default=None,
            help="最终输出目录",
            metavar="/tmp/output",
        )
        parser.add_argument(
            '-c',
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
    def __init__(self):
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

    def is_supported_image(self, file):
        ext = os.path.splitext(file)[1].lower()
        if ext == ".txt":
            return False
        return ext in Common.image_exts

    def paired_label(self, image):
        return f"{os.path.splitext(image)[0]}.txt"

    def negative_sample_count(self):
        train = self.args.train if self.args.train >= 0 else 0
        val = self.args.val if self.args.val >= 0 else 0
        return train, val

    def scan_negative_images(self):
        files = glob.glob(f"{self.args.source}/**/*", recursive=True)
        return sorted(
            [
                file
                for file in files
                if os.path.isfile(file) and self.is_supported_image(file)
            ]
        )

    def copy_negative_image(self, split, image):
        label = self.paired_label(image)
        if os.path.exists(label) and os.path.getsize(label) != 0:
            self.missed.append([os.path.relpath(image, self.args.source), "同名 .txt 不是 0 字节"])
            self.logger.warning(f"negative sample label not empty image={image} label={label}")
            return False

        relpath = os.path.relpath(image, self.args.source)
        relroot, ext = os.path.splitext(relpath)
        image_target = os.path.join(self.args.target, "images", split, f"{relroot}{ext}")
        label_target = os.path.join(self.args.target, "labels", split, f"{relroot}.txt")

        os.makedirs(os.path.dirname(image_target), exist_ok=True)
        os.makedirs(os.path.dirname(label_target), exist_ok=True)
        shutil.copy2(image, image_target)
        open(label_target, "w", encoding="utf-8").close()
        self.logger.info(f"negative sample split={split} image={image_target} label={label_target}")
        return True

    def negative_samples(self, args):
        self.args = args
        if not os.path.isdir(self.args.source):
            print(f"source 目录不存在: {self.args.source}")
            self.logger.error(f"source 目录不存在: {self.args.source}")
            exit()

        train_count, val_count = self.negative_sample_count()
        images = self.scan_negative_images()
        selected_train = images[:train_count] if train_count > 0 else []
        selected_val = images[train_count:train_count + val_count] if val_count > 0 else []

        total = len(selected_train) + len(selected_val)
        done = {"train": 0, "val": 0}
        with tqdm(total=total, ncols=120) as progress:
            for split, selected in (("train", selected_train), ("val", selected_val)):
                for image in selected:
                    progress.set_description(f"{split}/{os.path.relpath(image, self.args.source)}")
                    if self.copy_negative_image(split, image):
                        done[split] += 1
                    progress.update(1)

        if self.missed:
            table = Texttable(max_width=160)
            table.add_rows([["文件", "原因"], *self.missed])
            print(table.draw())

        table = Texttable(max_width=100)
        table.add_rows(
            [
                ["输出", "数量"],
                ["source images", len(images)],
                ["train", done["train"]],
                ["val", done["val"]],
                ["skip", len(self.missed)],
            ]
        )
        print(table.draw())

    def main(self, args):

        self.args = args

        if self.args.source and self.args.target:
            self.logger.info("Start")
            self.input()
            self.process()
            self.output()
            self.logger.info("Done")


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

    def __init__(self):

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

        classes = os.path.join(self.args.source, "classes.txt")
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error("classes.txt 文件不存在！")
            exit()

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

    def main(self, args):
        self.args = args
        if self.args.classes and self.args.source:
            self.classes()
        elif self.args.source and self.args.total:
            self.total()
        elif self.args.source and self.args.index:
            self.total()
        elif self.args.source and self.args.find:
            self.search()

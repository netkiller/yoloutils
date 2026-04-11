#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################
# Home	: https://www.netkiller.cn
# Author: Neo <netkiller@msn.com>
# Upgrade: 2025-03-29
# Description：YOLO 标签处理工具
# （标签删除/合并/修改/复制/图片尺寸/Labelimg2yolo）
##############################################

try:
    import argparse
    import glob
    import logging
    import os
    import random
    import shutil
    import sys
    import uuid
    from datetime import datetime
    import cv2
    import yaml
    from PIL import Image, ImageOps
    from texttable import Texttable
    from tqdm import tqdm
    from ultralytics import YOLO

except ImportError as err:
    print("Import Error: %s" % (err))
    exit()

try:
    from . import BASE_DIR, Common
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    from __init__ import BASE_DIR, Common

try:
    from .test import YoloTest
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from test import YoloTest
    else:
        raise

try:
    from .image import YoloImageCrop, YoloImageResize
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from image import YoloImageCrop, YoloImageResize
    else:
        raise

try:
    from .classify import YoloClassify
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from classify import YoloClassify
    else:
        raise

try:
    from .label import (
        YoloLabel,
        YoloLabelChange,
        YoloLabelCopy,
        YoloLabelMerge,
        YoloLabelRemove,
    )
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from label import (
            YoloLabel,
            YoloLabelChange,
            YoloLabelCopy,
            YoloLabelMerge,
            YoloLabelRemove,
        )
    else:
        raise


class YoloUtils:
    def __init__(self):
        # self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(self.basedir)

        # 日志记录基本设置
        # logfile = os.path.join(self.basedir, 'logs', f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d.%H%M%S')}.log")
        # logfile = os.path.join(self.basedir, 'logs', f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d')}.log")
        logfile = f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d')}.log"
        logging.basicConfig(
            filename=logfile,
            level=logging.DEBUG,
            encoding="utf-8",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        parser = argparse.ArgumentParser(
            description="Yolo 标签工具",
            epilog="Author: netkiller - https://www.netkiller.cn",
        )
        self.subparsers = parser.add_subparsers(
            title="subcommands",
            description="valid subcommands",
            dest="subcommand",
            help="additional help",
        )

        self.parent_parser = argparse.ArgumentParser(add_help=False)
        # parent_parser.add_argument('--parent', type=int)
        common = self.parent_parser.add_argument_group(
            title="通用参数", description=None
        )
        common.add_argument("--source", type=str, default=None, help="图片来源地址")
        common.add_argument("--target", default=None, type=str, help="图片目标地址")
        common.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )

        self.label = self.subparsers.add_parser("label", help="标签统计、索引统计、标签搜索")
        self.label.add_argument(
            "--source", type=str, default=None, help="目录", metavar="/tmp/dir1"
        )
        self.label.add_argument(
            "--classes",
            action="store_true",
            default=False,
            help="查看 classes.txt 文件",
        )
        self.label.add_argument(
            "--total", action="store_true", default=False, help="统计标签图数量"
        )
        self.label.add_argument(
            "--index", action="store_true", default=False, help="统计标签索引数量"
        )
        self.label.add_argument(
            "--search", nargs="+", default=None, help="搜索标签", metavar="1 2 3"
        )

        # labelimg.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

        self.merge = self.subparsers.add_parser(
            "merge", help="合并两个TXT文件中的标签到新TXT文件"
        )
        # self.parser = argparse.ArgumentParser(description='合并YOLO标签工具')
        self.merge.add_argument(
            "--left", type=str, default=None, help="左侧目录", metavar="/tmp/dir1"
        )
        self.merge.add_argument(
            "--right", default=None, type=str, help="右侧目录", metavar="/tmp/dir2"
        )
        self.merge.add_argument(
            "--output",
            type=str,
            default=None,
            help="最终输出目录",
            metavar="/tmp/output",
        )
        self.merge.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )

        # subparsers = self.parser.add_subparsers(help='subcommand help')

        self.copy = self.subparsers.add_parser("copy", help="从指定标签复制图片文件")
        self.copy.add_argument("--source", type=str, default=None, help="图片来源地址")
        self.copy.add_argument("--target", type=str, default=None, help="图片目标地址")
        self.copy.add_argument(
            "--label", type=str, default=None, help="逗号分割多个标签"
        )
        self.copy.add_argument(
            "-u", "--uuid", action="store_true", default=False, help="UUID 文件名"
        )
        self.copy.add_argument(
            "-c", "--clean", action="store_true", default=False, help="清理目标文件夹"
        )

        self.remove = self.subparsers.add_parser(
            "remove", help="从YOLO TXT文件中删除指定标签", parents=[self.parent_parser]
        )
        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')
        self.remove.add_argument(
            "--classes", nargs="+", default=None, help="标签序号", metavar="1 2 3"
        )
        self.remove.add_argument(
            "--label", nargs="+", default=None, help="标签名称", metavar="label1 label2"
        )
        # remove.add_argument('--output', type=str, default=None, help='输出目录', metavar="/tmp/output")
        # self.remove.add_argument('--clean', action="store_true", default=False, help='清理输出目录')
        # self.remove.add_argument('--show', action='store_true', help='查看 classes.txt 文件')

        self.change = self.subparsers.add_parser("change", help="修改标签索引")
        self.change.add_argument(
            "--source", type=str, default=None, help="目录", metavar="/tmp/dir1"
        )
        self.change.add_argument(
            "--search", nargs="+", default=None, help="标签序号", metavar="1 2 3"
        )
        self.change.add_argument(
            "--replace", nargs="+", default=None, help="标签名称", metavar="4 5 6"
        )

        self.crop = self.subparsers.add_parser(
            "crop", help="图片裁剪", parents=[self.parent_parser]
        )
        self.crop.add_argument(
            "--model", type=str, default=None, metavar="best.pt", help="模型"
        )
        self.crop.add_argument(
            "--output",
            type=str,
            default=None,
            help="Yolo 输出目录",
            metavar="/tmp/output",
        )
        # self.change.add_argument('--classes', action="store_true", default=False, help='查看 classes.txt 文件')
        # parser_b.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')
        #
        # # parse some argument lists
        # parser.parse_args(['a', '12'])
        # Namespace(bar=12, foo=False)

        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')
        # self.parser.add_argument('--label', type=int, default=-1, help='长边尺寸',metavar=0)

        # self.args = self.parser.parse_args()
        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')
        # self.parser.add_argument('--label', type=int, default=-1, help='长边尺寸',metavar=0)
        # self.parser = argparse.ArgumentParser(
        #     description='Yolo 工具 V3.0 - Design by netkiller - https://www.netkiller.cn')
        # self.parser.add_argument('--source', type=str, default=None, help='图片来源地址')
        # self.parser.add_argument('--target', default=None, type=str, help='图片目标地址')
        # self.parser.add_argument('--classes', type=str, default=None, help='classes.txt 文件')
        # self.parser.add_argument('--val', type=int, default=10, help='检验数量', metavar=10)
        # self.parser.add_argument('--crop', action="store_true", default=False, help='裁剪')
        # self.args = self.parser.parse_args()

        self.labelimg = self.subparsers.add_parser(
            "labelimg",
            help="labelimg 格式转换为 yolo 训练数据集",
            parents=[self.parent_parser],
        )
        # self.labelimg.add_argument('--source', type=str, default=None, help='图片来源地址')
        # self.labelimg.add_argument('--target', default=None, type=str, help='图片目标地址')
        self.labelimg.add_argument(
            "--classes", type=str, default=None, help="classes.txt 文件"
        )
        self.labelimg.add_argument(
            "--val", type=int, default=10, help="检验数量", metavar=10
        )
        # self.labelimg.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.labelimg.add_argument('--crop', action="store_true", default=False, help='裁剪')
        self.labelimg.add_argument(
            "--uuid", action="store_true", default=False, help="输出文件名使用UUID"
        )
        self.labelimg.add_argument(
            "--check",
            action="store_true",
            default=False,
            help="图片检查 corrupt JPEG restored and saved",
        )

        self.resize = self.subparsers.add_parser(
            "resize", help="修改图片尺寸", parents=[self.parent_parser]
        )
        # self.parser = argparse.ArgumentParser(description='自动切割学习数据')
        # self.resize.add_argument('--source', type=str, default=None, help='图片来源地址')
        self.resize.add_argument(
            "--imgsz", type=int, default=640, help="长边尺寸", metavar=640
        )
        self.resize.add_argument(
            "--output", type=str, default=None, help="输出识别图像", metavar=""
        )
        # self.resize.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.resize.add_argument('--md5sum', action="store_true", default=False, help='使用md5作为文件名')


        # self.args = self.parser.parse_args()

        self.classify = self.subparsers.add_parser(
            "classify", help="图像分类数据处理", parents=[self.parent_parser]
        )
        self.classify.add_argument(
            "--output", type=str, default=None, help="输出识别图像", metavar=""
        )
        self.classify.add_argument(
            "--checklist", type=str, default=None, help="输出识别图像", metavar=""
        )
        self.classify.add_argument(
            "--test", type=int, default=10, help="测试数量", metavar=100
        )
        # self.classify.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        self.classify.add_argument(
            "--crop", action="store_true", default=False, help="裁剪"
        )
        self.classify.add_argument(
            "--model", type=str, default=None, help="裁剪模型", metavar=""
        )
        self.classify.add_argument(
            "--uuid", action="store_true", default=False, help="重命名图片为UUID"
        )
        self.classify.add_argument(
            "--verbose", action="store_true", default=False, help="过程输出"
        )
        # ---------- 测试 ----------
        self.test = self.subparsers.add_parser(
            "test", help="模型测试工具", parents=[self.parent_parser]
        )
        self.test.add_argument('--model', type=str, default=None, help='模型路径')
        self.test.add_argument('--csv', type=str,default=None,  help='保存测试结果',metavar="result.csv")
        self.test.add_argument('--output', type=str,default=None,  help='测试结果输出路径')
        testGroup = self.test.add_argument_group(            title="对比模型", description="对比多个模型识别率")
        testGroup.add_argument('--diff', action="store_true", default=False, help='对比模型')
        testGroup.add_argument("--models", nargs = "+", default = None, help = "模型", metavar = "best1.pt best2.pt best3.pt")
        testGroup.add_argument('-l', '--label', type=str,default=None, help='标签统计',metavar="")

        self.parser = parser

    def main(self):

        args = self.parser.parse_args()

        # print(args, args.subcommand)
        if args.subcommand == "label":
            run = YoloLabel(self.label, args)
        elif args.subcommand == "copy":
            run = YoloLabelCopy(self.copy, args)
        elif args.subcommand == "remove":
            run = YoloLabelRemove(self.remove, args)
        elif args.subcommand == "change":
            run = YoloLabelChange(self.change, args)
        elif args.subcommand == "merge":
            run = YoloLabelMerge(self.merge, args)
        elif args.subcommand == "labelimg":
            run = YoloLabelimg(self.labelimg, args)
        elif args.subcommand == "resize":
            run = YoloImageResize(self.resize, args)
        elif args.subcommand == "crop":
            run = YoloImageCrop(self.crop, args)
        elif args.subcommand == "test":
            run = YoloTest(self.test, args)
        elif args.subcommand == "classify":
            run = YoloClassify(self.classify, args)
        else:
            self.parser.print_help()
            exit()

        run.main()



class YoloLabelimg(Common):
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)

    def __init__(self, parser, args):
        self.basedir = BASE_DIR
        # print(self.basedir)
        # print(logfile)
        # sys.path.append(self.basedir)

        # 日志记录基本设置
        logfile = os.path.join(
            self.basedir,
            "logs",
            f"{os.path.splitext(os.path.basename(__file__))[0]}.log",
        )
        logging.basicConfig(
            filename=logfile,
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.parser = parser
        self.args = args

        self.classes = []
        self.lables = {}
        self.missed = []

        self.logger = logging.getLogger("LabelimgToYolo")

    def mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def input(self):
        if self.args.clean:
            if os.path.exists(self.args.target):
                shutil.rmtree(self.args.target)

        self.mkdirs(os.path.join(self.args.target))
        directory = [
            "train/labels",
            "train/images",
            "val/labels",
            "val/images",
            "test/labels",
            "test/images",
        ]

        classes = os.path.join(self.args.source, "classes.txt")
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error(f"classes.txt 文件不存在！")
            exit()
        else:
            with open(classes) as file:
                for line in file:
                    self.classes.append(line.strip())
                    self.lables[line.strip()] = []
                self.logger.info(
                    f"classes len={len(self.classes)} labels={self.classes}"
                )

        with tqdm(total=len(directory), ncols=120) as progress:
            for dir in directory:
                progress.set_description(f"init {dir}")
                self.mkdirs(os.path.join(self.args.target, dir))
                progress.update(1)

    def process(self):
        # images =  glob.glob('*.jpg', root_dir=self.args.source)
        # labels = glob.glob('*.txt', root_dir=self.args.source)
        files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)

        with (
            tqdm(total=len(files), ncols=150) as images,
            tqdm(total=len(files), ncols=150) as train,
        ):
            for source in files:
                if source.endswith("classes.txt"):
                    train.update(1)
                    images.update(1)
                    continue
                train.set_description(f"train/labels: {source}")

                uuid4 = None
                if self.args.uuid:
                    uuid4 = uuid.uuid4()
                    target = os.path.join(
                        self.args.target, "train/labels", f"{uuid4}.txt"
                    )
                else:
                    target = os.path.join(
                        self.args.target, "train/labels", os.path.basename(source)
                    )
                name, extension = os.path.splitext(os.path.basename(target))

                with open(source) as file:
                    lines = []
                    for line in file:
                        index = line.strip().split(" ")[0]
                        try:
                            label = self.classes[int(index)]
                            # if label not in self.lables:
                            #     self.lables[label] = []
                            self.lables[label].append(name)
                            lines.append(label)
                        # self.logger.debug(f"label={label} count={len(self.lables[label])} index={index} file={name} line={line.strip()} ")
                        except IndexError as e:
                            self.logger.error(f"{repr(e)}, {index}")
                    self.logger.info(f"file={name} labels={lines}")

                shutil.copy(source, target)
                self.logger.debug(
                    f"train/labels source={source} target={target} name={name}"
                )
                train.update(1)
                images.set_description(f"train/images: {source}")

                for ext in [".jpg", ".png"]:
                    source = source.replace(".txt", ext)
                    if os.path.exists(source):
                        target = os.path.join(
                            self.args.target, "train/images", f"{name}.jpg"
                        )
                        shutil.copy(source, target)
                        self.logger.info(
                            f"train/images source={source} target={target} name={name}"
                        )
                    else:
                        self.logger.warning(
                            f"train/images source={source} target={target} name={name}"
                        )
                    break
                images.update(1)

        for label, files in self.lables.items():
            if len(files) == 0:
                continue
            if len(files) < self.args.val:
                valnumber = len(files)
            else:
                valnumber = self.args.val

            vals = random.sample(files, valnumber)
            # print(f"label={label} files={len(files)} val={len(vals)}")

            with tqdm(total=len(vals), ncols=120) as progress:
                for file in vals:
                    progress.set_description(f"val/label {label}")
                    name, extension = os.path.splitext(os.path.basename(file))
                    try:
                        source = os.path.join(
                            self.args.target, "train/labels", f"{name}.txt"
                        )
                        target = os.path.join(
                            self.args.target, "val/labels", f"{name}.txt"
                        )
                        if os.path.exists(target):
                            self.logger.info(
                                f"val/labels skip label={label} file={file}"
                            )
                            progress.update(1)
                            continue

                        shutil.copy(source, target)
                        self.logger.info(
                            f"val/labels copy label={label} source={source} target={target}"
                        )

                        source = os.path.join(
                            self.args.target, "train/images", f"{name}.jpg"
                        )
                        target = os.path.join(
                            self.args.target, "val/images", f"{name}.jpg"
                        )
                        shutil.copy(source, target)
                        self.logger.info(
                            f"val/images copy label={label} source={source} target={target}"
                        )
                    except Exception as e:
                        self.logger.error(f"val {repr(e)} name={name}")
                    progress.update(1)

    def output(self):
        names = {i: self.classes[i] for i in range(len(self.classes))}  # 标签类别
        data = {
            "path": os.path.join(os.getcwd(), self.args.target),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": names,
            # 'nc': len(self.classes)
        }
        with open(
            os.path.join(self.args.target, "data.yaml"), "w", encoding="utf-8"
        ) as file:
            yaml.dump(data, file, allow_unicode=True)

    def report(self):
        tables = [["标签", "数量"]]
        for label, files in self.lables.items():
            # if len(files) == 0:
            #     continue
            tables.append([label, len(files)])
        table = Texttable(max_width=160)
        table.add_rows(tables)
        print(table.draw())
        for file in self.missed:
            self.logger.warning(f"丢失文件 {file}")

    def main(self):

        if self.args.source and self.args.target:
            self.logger.info("Start")
            self.input()
            self.process()
            self.output()
            self.report()
            self.logger.info("Done")
        else:
            self.parser.parse_args(["labelimg"])

            self.parser.print_help()
            exit()



def main():
    try:
        run = YoloUtils()
        run.main()
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()

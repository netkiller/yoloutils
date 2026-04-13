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
    from .test import YoloTest, YoloTestDiff
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from test import YoloTest, YoloTestDiff
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

try:
    from .labelimg import YoloLabelimg, YoloLabelimgAutomatic
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from labelimg import YoloLabelimg, YoloLabelimgAutomatic
    else:
        raise


class YoloUtils:
    def __init__(self):
        nowrap_formatter = lambda prog: argparse.HelpFormatter(
            prog, max_help_position=32, width=4096
        )

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
            description="Yolo 标签与图像处理工具",
            epilog="Author: netkiller - https://www.netkiller.cn",
        )
        self.subparsers = parser.add_subparsers(
            title="子命令",
            description="工具含标签类处理和图像类处理工具",
            dest="subcommand",
            help="风险提示：当使用 --clean 参数时会删除目标目录和输出目录 ",
        )

        self.label = self.subparsers.add_parser("label", help="标签统计、索引统计、标签搜索")

        # labelimg.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

        self.merge = self.subparsers.add_parser(
            "merge", help="合并两个TXT文件中的标签到新TXT文件"
        )
        # self.parser = argparse.ArgumentParser(description='合并YOLO标签工具')

        # subparsers = self.parser.add_subparsers(help='subcommand help')

        self.copy = self.subparsers.add_parser("copy", help="从指定标签复制图片文件")
        self.remove = self.subparsers.add_parser("remove", help="从YOLO TXT文件中删除指定标签")
        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')

        # remove.add_argument('--output', type=str, default=None, help='输出目录', metavar="/tmp/output")
        # self.remove.add_argument('--clean', action="store_true", default=False, help='清理输出目录')
        # self.remove.add_argument('--show', action='store_true', help='查看 classes.txt 文件')

        self.change = self.subparsers.add_parser("change", help="修改标签索引")
        self.crop = self.subparsers.add_parser("crop", help="图片裁剪")

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

        # self.parser.add_argument('--crop', action="store_true", default=False, help='裁剪')
        # self.args = self.parser.parse_args()

        self.labelimg = self.subparsers.add_parser(
            "labelimg",
            help="labelimg 格式转换为 yolo 训练数据集",
            epilog="将 labelimg 标注的图像转换为 yolo 训练数据集",
            formatter_class=nowrap_formatter,
        )

        self.auto = self.subparsers.add_parser(
            "auto",
            help="用现有模型自动给训练图像打标签",
            epilog="用载入的模型自动给目录中的文件打标",
            formatter_class=nowrap_formatter,
        )

        self.resize = self.subparsers.add_parser("resize", help="修改图片尺寸")
        # self.parser = argparse.ArgumentParser(description='自动切割学习数据')
        # self.resize.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.resize.add_argument('--md5sum', action="store_true", default=False, help='使用md5作为文件名')
        # self.args = self.parser.parse_args()

        self.classify = self.subparsers.add_parser("classify", help="图像分类数据处理")

        # ---------- 测试 ----------
        self.test = self.subparsers.add_parser("test", help="模型测试工具")
        self.diff = self.subparsers.add_parser("diff", help="模型比较工具")

        self.parser = parser

    def main(self):
        argv = sys.argv[1:]
        try:
            if not argv:
                self.parser.print_help()
                return
            root_args = self.parser.parse_args([argv[0]])
        except SystemExit as e:
            if e.code != 0:
                self.parser.print_help(sys.stderr)
            raise

        run = None
        if root_args.subcommand == "label":
            run = YoloLabel(self.label, root_args)
        elif root_args.subcommand == "copy":
            run = YoloLabelCopy(self.copy, root_args)
        elif root_args.subcommand == "remove":
            run = YoloLabelRemove(self.remove, root_args)
        elif root_args.subcommand == "change":
            run = YoloLabelChange(self.change, root_args)
        elif root_args.subcommand == "merge":
            run = YoloLabelMerge(self.merge, root_args)
        elif root_args.subcommand == "labelimg":
            run = YoloLabelimg(self.labelimg, root_args)
        elif root_args.subcommand == "auto":
            run = YoloLabelimgAutomatic(self.auto, root_args)
        elif root_args.subcommand == "resize":
            run = YoloImageResize(self.resize, root_args)
        elif root_args.subcommand == "crop":
            run = YoloImageCrop(self.crop, root_args)
        elif root_args.subcommand == "test":
            run = YoloTest(self.test, root_args)
        elif root_args.subcommand == "diff":
            run = YoloTestDiff(self.diff, root_args)
        elif root_args.subcommand == "classify":
            run = YoloClassify(self.classify, root_args)

        if run is None:
            self.parser.print_help()
            exit()

        try:
            sub_args = run.parser.parse_args(argv[1:])
        except SystemExit as e:
            if e.code != 0:
                run.parser.print_help(sys.stderr)
            raise

        setattr(sub_args, "subcommand", root_args.subcommand)
        run.args = sub_args
        run.main()


def main():
    try:
        run = YoloUtils()
        run.main()
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()

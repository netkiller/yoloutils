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

try:
    from .info import YoloInfo
except ImportError:
    # Support direct script execution (python yoloutils.py ...)
    if __name__ == "__main__":
        from info import YoloInfo
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

        self.info = self.subparsers.add_parser(name='info', help='查看模型信息')
        self.info.add_argument('-m', '--model', type=str, default=None, help='模型路径')
        self.info.add_argument('-i', '--info', action="store_true", default=False, help='模型信息')
        self.info.add_argument('-r', '--recall', action="store_true", default=False, help='召回率')
        self.info.add_argument('-p', '--precision', action="store_true", default=False, help='精确率')
        self.info.add_argument('-a', '--accuracy', action="store_true", default=False, help='准确率')
        self.info.add_argument('--f1', action="store_true", default=False, help='输出 F1 指标')
        self.info.add_argument('--mAP', action="store_true", default=False, help='平均精度均值 mAP（mean Average Precision）')

        self.label = self.subparsers.add_parser("label", help="标签统计、索引统计、标签搜索")
        self.label.add_argument('-s',
                                "--source", type=str, default=None, help="目录", metavar="/tmp/dir1"
                                )
        self.label.add_argument(
            '-c',
            "--classes",
            action="store_true",
            default=False,
            help="查看 classes.txt 文件",
        )
        self.label.add_argument(
            '-t',
            "--total", action="store_true", default=False, help="统计标签图数量"
        )
        self.label.add_argument(
            '-i',
            "--index", action="store_true", default=False, help="统计标签索引数量"
        )
        self.label.add_argument(
            '-f',
            "--find", nargs="+", default=None, help="搜索标签", metavar="1 2 3"
        )

        # labelimg.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

        self.merge = self.subparsers.add_parser(
            "merge", help="合并两个TXT文件中的标签到新TXT文件"
        )
        # self.parser = argparse.ArgumentParser(description='合并YOLO标签工具')

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

        # self.args = self.parser.parse_args()
        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')
        # self.parser.add_argument('--label', type=int, default=-1, help='长边尺寸',metavar=0)
        # self.parser = argparse.ArgumentParser(
        #     description='Yolo 工具 V3.0 - Design by netkiller - https://www.netkiller.cn')

        # self.args = self.parser.parse_args()

        self.labelimg = self.subparsers.add_parser(
            "labelimg",
            help="labelimg 格式转换为 yolo 训练数据集",
            epilog="将 labelimg 标注的图像转换为 yolo 训练数据集",
            formatter_class=nowrap_formatter,
        )
        self.labelimg.add_argument('-s', "--source", type=str, default=None, help="图片来源地址")
        self.labelimg.add_argument('-t', "--target", type=str, default=None, help="图片目标地址")
        self.labelimg.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        self.labelimg.add_argument(
            '-c', "--classes", type=str, default=None, help="classes.txt 文件"
        )
        self.labelimg.add_argument(
            '-v', "--val", type=int, default=10, help="验证集占比（百分比 10% ~ 80%）", metavar=10
        )
        # self.labelimg.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')

        self.labelimg.add_argument(
            '-u', "--uuid", action="store_true", default=False, help="输出文件名使用UUID"
        )
        # self.labelimg.add_argument(
        #     "--check",
        #     action="store_true",
        #     default=False,
        #     help="图片检查 corrupt JPEG restored and saved",
        # )
        self.labelimg.add_argument('-r', '--report', type=str, default=None, help='输出 csv 报告')

        self.auto = self.subparsers.add_parser(
            "auto",
            help="用现有模型自动给训练图像打标签",
            epilog="用载入的模型自动给目录中的文件打标",
            formatter_class=nowrap_formatter,
        )
        self.auto.add_argument("--source", type=str, default=None, help="图片来源地址")
        self.auto.add_argument("--target", type=str, default=None, help="图片目标地址")
        self.auto.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )

        self.auto.add_argument('--model', type=str, default=None, help='载入模型', metavar="best.pt")
        self.auto.add_argument('--conf', type=float, default=None, help='置信度阈值', metavar=0.5)
        self.auto.add_argument('--csv', default=None, type=str, help='报告输出，哪些文件已经标准，哪些没有标注', metavar="report.csv")
        self.auto.add_argument('--output', type=str, default=None, help='输出标注效果', metavar="/path/to/output")

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
        # print(self.parser.parse_args())

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
            try:
                sub_args = self.label.parse_args(argv[1:])
                has_action = any(
                    [
                        sub_args.classes,
                        sub_args.total,
                        sub_args.index,
                        sub_args.find,
                    ]
                )
                if sub_args.source and has_action:
                    run = YoloLabel()
                    run.main(sub_args)
                else:
                    self.label.print_help()

            except SystemExit as e:
                if e.code != 0:
                    self.label.print_help(sys.stderr)
                raise
            exit()
        elif root_args.subcommand == "copy":
            run = YoloLabelCopy(self.copy, root_args)
        elif root_args.subcommand == "remove":
            run = YoloLabelRemove(self.remove, root_args)
        elif root_args.subcommand == "change":
            run = YoloLabelChange(self.change, root_args)
        elif root_args.subcommand == "merge":
            run = YoloLabelMerge(self.merge, root_args)
        elif root_args.subcommand == "labelimg":
            try:
                sub_args = self.labelimg.parse_args(argv[1:])
            except SystemExit as e:
                if e.code != 0:
                    self.labelimg.print_help(sys.stderr)
                raise

            if sub_args.source and sub_args.target:
                run = YoloLabelimg()
                run.main(sub_args)
                exit()

            self.labelimg.print_help()
            exit()

        elif root_args.subcommand == "auto":
            try:
                sub_args = self.auto.parse_args(argv[1:])
            except SystemExit as e:
                if e.code != 0:
                    self.auto.print_help(sys.stderr)
                raise

            if sub_args.source and sub_args.target and sub_args.model:
                run = YoloLabelimgAutomatic()
                run.main(sub_args)
                exit()
            else:
                self.auto.print_help()
            exit()

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
        elif root_args.subcommand == "info":

            try:
                sub_args = self.info.parse_args(argv[1:])
            except SystemExit as e:
                if e.code != 0:
                    self.info.print_help(sys.stderr)
                raise
            model = sub_args.model or getattr(sub_args, "target", None)
            if not model:
                self.info.print_help()
                exit()
            run = YoloInfo(model=model)
            if sub_args.f1:
                run.f1()
                exit()
            if sub_args.recall:
                run.recall()
                exit()
            if sub_args.precision:
                run.precision()
                exit()
            if sub_args.accuracy:
                run.accuracy()
                exit()
            if sub_args.mAP:
                run.mAP()
                exit()
            if sub_args.info:
                run.info()
                exit()
            else:
                self.info.print_help()
            exit()

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
        # print(e)
        pass


if __name__ == "__main__":
    main()

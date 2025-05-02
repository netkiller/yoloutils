#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################
# Home	: https://www.netkiller.cn
# Author: Neo <netkiller@msn.com>
# Upgrade: 2025-03-29
# YOLO 标签处理工具：
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
    import hashlib
    import yaml
    import cv2
    from datetime import datetime
    from PIL import Image, ImageOps
    from texttable import Texttable
    from tqdm import tqdm

    from ultralytics import YOLO

except ImportError as err:
    print("Import Error: %s" % (err))
    exit()


class YoloUtils():
    def __init__(self):
        # self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(self.basedir)

        # 日志记录基本设置
        # logfile = os.path.join(self.basedir, 'logs', f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d.%H%M%S')}.log")
        # logfile = os.path.join(self.basedir, 'logs', f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d')}.log")
        logfile = f"{os.path.splitext(os.path.basename(__file__))[0]}.{datetime.today().strftime('%Y-%m-%d')}.log"
        logging.basicConfig(filename=logfile, level=logging.DEBUG, encoding="utf-8",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        parser = argparse.ArgumentParser(description='Yolo 标签工具',
                                         epilog='Author: netkiller - https://www.netkiller.cn')
        self.subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', dest='subcommand',
                                                help='additional help')

        self.parent_parser = argparse.ArgumentParser(add_help=False)
        # parent_parser.add_argument('--parent', type=int)
        common = self.parent_parser.add_argument_group(title='通用参数', description=None)
        common.add_argument('--source', type=str, default=None, help='图片来源地址')
        common.add_argument('--target', default=None, type=str, help='图片目标地址')
        common.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')

        self.label = self.subparsers.add_parser('label', help='标签处理工具')
        self.label.add_argument('--source', type=str, default=None, help='目录', metavar="/tmp/dir1")
        self.label.add_argument('--classes', action="store_true", default=False, help='查看 classes.txt 文件')
        self.label.add_argument('--total', action="store_true", default=False, help='统计标签图数量')
        self.label.add_argument('--index', action="store_true", default=False, help='统计标签索引数量')
        self.label.add_argument('--search', nargs='+', default=None, help='搜索标签', metavar="1 2 3")

        # labelimg.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

        self.merge = self.subparsers.add_parser('merge', help='合并两个TXT文件中的标签到新TXT文件')
        # self.parser = argparse.ArgumentParser(description='合并YOLO标签工具')
        self.merge.add_argument('--left', type=str, default=None, help='左侧目录', metavar="/tmp/dir1")
        self.merge.add_argument('--right', default=None, type=str, help='右侧目录', metavar="/tmp/dir2")
        self.merge.add_argument('--output', type=str, default=None, help='最终输出目录', metavar="/tmp/output")
        self.merge.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')

        # subparsers = self.parser.add_subparsers(help='subcommand help')

        self.copy = self.subparsers.add_parser('copy', help='从指定标签复制图片文件')
        self.copy.add_argument('--source', type=str, default=None, help='图片来源地址')
        self.copy.add_argument('--target', type=str, default=None, help='图片目标地址')
        self.copy.add_argument('--label', type=str, default=None, help='逗号分割多个标签')
        self.copy.add_argument('-u', '--uuid', action="store_true", default=False, help='UUID 文件名')
        self.copy.add_argument('-c', '--clean', action="store_true", default=False, help='清理目标文件夹')

        self.remove = self.subparsers.add_parser('remove', help='从YOLO TXT文件中删除指定标签',
                                                 parents=[self.parent_parser])
        # self.parser = argparse.ArgumentParser(description='YOLO标签删除工具')
        self.remove.add_argument('--classes', nargs='+', default=None, help='标签序号', metavar="1 2 3")
        self.remove.add_argument('--label', nargs='+', default=None, help='标签名称', metavar="label1 label2")
        # remove.add_argument('--output', type=str, default=None, help='输出目录', metavar="/tmp/output")
        # self.remove.add_argument('--clean', action="store_true", default=False, help='清理输出目录')
        # self.remove.add_argument('--show', action='store_true', help='查看 classes.txt 文件')

        self.change = self.subparsers.add_parser('change', help='修改标签索引')
        self.change.add_argument('--source', type=str, default=None, help='目录', metavar="/tmp/dir1")
        self.change.add_argument('--search', nargs='+', default=None, help='标签序号', metavar="1 2 3")
        self.change.add_argument('--replace', nargs='+', default=None, help='标签名称', metavar="4 5 6")

        self.crop = self.subparsers.add_parser('crop', help='图片裁剪', parents=[self.parent_parser])
        self.crop.add_argument('--model', type=str, default=None, metavar="best.pt", help='模型')
        self.crop.add_argument('--output', type=str, default=None, help='Yolo 输出目录', metavar="/tmp/output")
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

        self.labelimg = self.subparsers.add_parser('labelimg', help='labelimg 格式转换为 yolo 训练数据集',
                                                   parents=[self.parent_parser])
        # self.labelimg.add_argument('--source', type=str, default=None, help='图片来源地址')
        # self.labelimg.add_argument('--target', default=None, type=str, help='图片目标地址')
        self.labelimg.add_argument('--classes', type=str, default=None, help='classes.txt 文件')
        self.labelimg.add_argument('--val', type=int, default=10, help='检验数量', metavar=10)
        # self.labelimg.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.labelimg.add_argument('--crop', action="store_true", default=False, help='裁剪')
        self.labelimg.add_argument('--uuid', action="store_true", default=False, help='输出文件名使用UUID')
        self.labelimg.add_argument('--check', action="store_true", default=False,
                                   help='图片检查 corrupt JPEG restored and saved')
        # self.labelimg.add_argument('-l', '--label', action="store_true", default=False, help='标签统计')

        self.resize = self.subparsers.add_parser('resize', help='修改图片尺寸', parents=[self.parent_parser])
        # self.parser = argparse.ArgumentParser(description='自动切割学习数据')
        # self.resize.add_argument('--source', type=str, default=None, help='图片来源地址')
        # self.resize.add_argument('--target', default=None, type=str, help='图片目标地址')
        self.resize.add_argument('--imgsz', type=int, default=640, help='长边尺寸', metavar=640)
        self.resize.add_argument('--output', type=str, default=None, help='输出识别图像', metavar="")
        # self.resize.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.resize.add_argument('--md5sum', action="store_true", default=False, help='使用md5作为文件名')
        # self.resize.add_argument('--uuid', action="store_true", default=False, help='重命名图片为UUID')
        # self.resize.add_argument('--crop', action="store_true", default=False, help='裁剪')
        # self.args = self.parser.parse_args()

        self.classify = self.subparsers.add_parser('classify', help='图像分类数据处理', parents=[self.parent_parser])
        # self.classify.add_argument('--source', type=str, default=None, help='图片来源地址')
        # self.classify.add_argument('--target', default=None, type=str, help='图片目标地址')
        self.classify.add_argument('--output', type=str, default=None, help='输出识别图像', metavar="")
        self.classify.add_argument('--checklist', type=str, default=None, help='输出识别图像', metavar="")
        self.classify.add_argument('--test', type=int, default=10, help='测试数量', metavar=100)
        # self.classify.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        self.classify.add_argument('--crop', action="store_true", default=False, help='裁剪')
        self.classify.add_argument('--model', type=str, default=None, help='裁剪模型', metavar="")
        self.classify.add_argument('--uuid', action="store_true", default=False, help='重命名图片为UUID')
        self.classify.add_argument('--verbose', action="store_true", default=False, help='过程输出')

        self.parser = parser

    def main(self):

        args = self.parser.parse_args()

        # print(args, args.subcommand)
        if args.subcommand == 'label':
            run = YoloLabel(self.label, args)
        elif args.subcommand == 'copy':
            run = YoloLabelCopy(self.copy, args)
        elif args.subcommand == 'remove':
            run = YoloLabelRemove(self.remove, args)
        elif args.subcommand == 'change':
            run = YoloLabelChange(self.change, args)
        elif args.subcommand == 'change':
            run = YoloLabelMerge(self.change, args)
        elif args.subcommand == 'labelimg':
            run = YoloLabelimg(self.labelimg, args)
        elif args.subcommand == 'resize':
            run = YoloResize(self.resize, args)
        elif args.subcommand == 'crop':
            run = ImageCrop(self.crop, args)
        elif args.subcommand == 'classify':
            run = Classify(self.classify, args)
        else:
            self.parser.print_help()
            exit()

        run.main()


class Common():
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)

    def __init__(self):
        self.logger = None
        self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(self.basedir)

    def mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def scanfile(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isfile(os.path.join(path, name)):
                files.append(name)

        return (files)

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)

        return (files)

    def walkdir(self, path):
        for dirpath, dirnames, filenames in os.walk(path):
            print(f"dirpath={dirpath}, dirnames={dirnames}, filenames={filenames}")
            # print(filenames)

    def md5sum(self, filename):
        md5 = hashlib.md5()
        with open(filename, 'rb') as f:
            md5.update(f.read())
            return md5.hexdigest()


class YoloLabelimg(Common):
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)

    def __init__(self, parser, args):
        self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # print(self.basedir)
        # print(logfile)
        # sys.path.append(self.basedir)

        # 日志记录基本设置
        logfile = os.path.join(self.basedir, 'logs', f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
        logging.basicConfig(filename=logfile, level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
            'train/labels', 'train/images', 'val/labels', 'val/images', 'test/labels', 'test/images'
        ]

        classes = os.path.join(self.args.source, 'classes.txt')
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error(f"classes.txt 文件不存在！")
            exit()
        else:
            with open(classes) as file:
                for line in file:
                    self.classes.append(line.strip())
                    self.lables[line.strip()] = []
                self.logger.info(f"classes len={len(self.classes)} labels={self.classes}")

        with tqdm(total=len(directory), ncols=120) as progress:
            for dir in directory:
                progress.set_description(f"init {dir}")
                self.mkdirs(os.path.join(self.args.target, dir))
                progress.update(1)

    def process(self):
        # images =  glob.glob('*.jpg', root_dir=self.args.source)
        # labels = glob.glob('*.txt', root_dir=self.args.source)
        files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)

        with tqdm(total=len(files), ncols=150) as images, tqdm(total=len(files), ncols=150) as train:

            for source in files:
                if source.endswith('classes.txt'):
                    train.update(1)
                    images.update(1)
                    continue
                train.set_description(f'train/labels: {source}')

                uuid4 = None
                if self.args.uuid:
                    uuid4 = uuid.uuid4()
                    target = os.path.join(self.args.target, 'train/labels', f"{uuid4}.txt")
                else:
                    target = os.path.join(self.args.target, 'train/labels', os.path.basename(source))
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
                self.logger.debug(f"train/labels source={source} target={target} name={name}")
                train.update(1)
                images.set_description(f'train/images: {source}')

                for ext in ['.jpg', '.png']:
                    source = source.replace('.txt', ext)
                    if os.path.exists(source):
                        target = os.path.join(self.args.target, 'train/images', f"{name}.jpg")
                        shutil.copy(source, target)
                        self.logger.info(f"train/images source={source} target={target} name={name}")
                    else:
                        self.logger.warning(f"train/images source={source} target={target} name={name}")
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
                        source = os.path.join(self.args.target, 'train/labels', f"{name}.txt")
                        target = os.path.join(self.args.target, 'val/labels', f"{name}.txt")
                        if os.path.exists(target):
                            self.logger.info(f"val/labels skip label={label} file={file}")
                            progress.update(1)
                            continue

                        shutil.copy(source, target)
                        self.logger.info(f"val/labels copy label={label} source={source} target={target}")

                        source = os.path.join(self.args.target, 'train/images', f"{name}.jpg")
                        target = os.path.join(self.args.target, 'val/images', f"{name}.jpg")
                        shutil.copy(source, target)
                        self.logger.info(f"val/images copy label={label} source={source} target={target}")
                    except Exception as e:
                        self.logger.error(f"val {repr(e)} name={name}")
                    progress.update(1)

    def output(self):
        names = {i: self.classes[i] for i in range(len(self.classes))}  # 标签类别
        data = {
            'path': os.path.join(os.getcwd(), self.args.target),
            'train': "train/images",
            'val': "val/images",
            'test': "test/images",
            'names': names
            # 'nc': len(self.classes)
        }
        with open(os.path.join(self.args.target, 'data.yaml'), 'w', encoding="utf-8") as file:
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

            self.parser.parse_args(['labelimg'])

            self.parser.print_help()
            exit()


class YoloLabelRemove(Common):
    total = {
        'change': 0,
        'remove': 0,
        'skip': 0,
        'error': 0
    }

    def __init__(self, parser, args):

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("remove")

        self.indexs = []

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)

        return (files)

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.target):
                    shutil.rmtree(self.args.target)
            if self.args.target:
                os.makedirs(self.args.target, exist_ok=True)

            # self.files = glob.glob(os.path.join(self.args.source, "*.txt"))
            self.files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)

            if self.args.label:
                classes = os.path.join(self.args.source, 'classes.txt')
                if not os.path.isfile(classes):
                    print(f"classes.txt 文件不存在: {classes}")
                    self.logger.error(f"classes.txt 文件不存在！")
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
            # print(self.files)
        except Exception as e:
            self.logger.error("input: ", repr(e))
            exit()

    def process(self):
        with tqdm(total=len(self.files), ncols=150) as progress:
            for file in self.files:
                progress.set_description(file)
                filename = os.path.basename(file)
                try:
                    if filename.lower() == 'classes.txt':
                        progress.update(1)
                        self.total['skip'] += 1
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
                                    # if line.startswith(f"{self.args.label} "):
                                    self.logger.info(f"index={index} indexs={self.indexs}")
                                    isChange = True
                                    continue
                                lines.append(line)
                        if len(lines) > 0:
                            if isChange:
                                with open(target, "w") as newfile:
                                    newfile.writelines(lines)
                                self.total['change'] += 1
                                self.logger.info(f"change target={target}")
                        else:
                            os.remove(target)
                            os.remove(target.replace(".txt", ".jpg"))
                            self.total['remove'] += 1
                            self.logger.info(f"remove target={target}")

                except FileNotFoundError as e:
                    self.logger.error(str(e))
                    self.total['error'] += 1

                progress.update(1)

    def output(self):
        tables = [["操作", "处理"]]
        tables.append(['count', len(self.files)])
        for k, v in self.total.items():
            tables.append([k, v])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())
        pass

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

        self.parser = parser
        self.args = args

        self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(self.basedir)
        # print(basedir)

        self.parser = argparse.ArgumentParser(description='合并YOLO标签工具')
        self.parser.add_argument('--left', type=str, default=None, help='左侧目录', metavar="/tmp/dir1")
        self.parser.add_argument('--right', default=None, type=str, help='右侧目录', metavar="/tmp/dir2")
        # self.parser.add_argument('--imgsz', type=int, default=640, help='长边尺寸',metavar=640)
        self.parser.add_argument('--output', type=str, default=None, help='最终输出目录', metavar="/tmp/output")
        self.parser.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # self.parser.add_argument('--md5sum', action="store_true", default=False, help='使用md5作为文件名')
        # self.parser.add_argument('--uuid', action="store_true", default=False, help='重命名图片为UUID')
        # self.parser.add_argument('--crop', action="store_true", default=False, help='裁剪')
        self.args = self.parser.parse_args()

    def scanfile(self, path):
        files = []
        # for name in os.listdir(path):
        #     if os.path.isfile(os.path.join(path, name)):
        #         files.append(name)
        files = glob.glob(path)

        return (files)

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)

        return (files)

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.output):
                    shutil.rmtree(self.args.output)
            os.makedirs(self.args.output, exist_ok=True)

            self.lefts = self.scanfile(os.path.join(self.args.left, "*.txt"))
            self.rights = self.scanfile(os.path.join(self.args.right, "*.txt"))
            # print(self.files)
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
                    if filename.lower() == 'classes.txt':
                        shutil.copyfile(file, os.path.join(self.args.output, filename))
                    else:
                        left = os.path.join(self.args.left, filename)
                        right = os.path.join(self.args.right, filename.replace('_0.', '.'))
                        output = os.path.join(self.args.output, filename)
                        image = filename.replace('.txt', '.jpg')
                        # print(f"left={left}, right={right}, output={output}")

                        shutil.copyfile(os.path.join(self.args.left, image), os.path.join(self.args.output, image))

                        if not os.path.isfile(right):
                            shutil.copyfile(left, output)
                            # print(f"test={os.path.isdir(right)} right={right}")
                        else:
                            with open(left, "r") as file1, open(right, "r") as file2, open(output, "w") as file:
                                txt1 = file1.read()
                                txt2 = file2.read()

                                file.write(txt1)
                                file.write(txt2)
                                # print(f"txt1={txt1}, txt2={txt2}")

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
        pass

    def main(self):
        # print(self.args)
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

        self.parser = parser
        self.args = args

        # print(self.args)

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

        classes = os.path.join(self.args.source, 'classes.txt')
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error(f"classes.txt 文件不存在！")
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
                for label in self.args.label.split(','):
                    if label in self.classes.keys():
                        self.lables.append(self.classes[label])
                    else:
                        self.logger.error(f"label {label} 不存在")
                        exit()
                self.logger.info(f"label len={len(self.lables)} list={self.lables}")

        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())

        self.files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)

    def process(self):

        with tqdm(total=len(self.files), ncols=150) as processBar, tqdm(total=len(self.files),
                                                                        ncols=150) as processBarImage:

            for file in self.files:

                processBar.set_description(f'{file}')

                if file.endswith('classes.txt'):
                    processBar.update(1)
                    processBarImage.update(1)
                    self.logger.info(f"skip classes.txt")
                    continue

                source = file
                if self.args.uuid:
                    uuid4 = uuid.uuid4()
                    target = os.path.join(self.args.target, f"{uuid4}.txt")
                else:
                    target = os.path.join(self.args.target, os.path.basename(source))

                image = file.replace('.txt', '.jpg', 1)
                processBarImage.set_description(f'{image}')

                if self.args.label:

                    with open(file) as txt:
                        # self.logger.debug(f"TXT {file} {txt.readlines()}")
                        for line in txt.readlines():
                            # self.logger.info(f">>> {line}")
                            index = int(line.strip().split(" ")[0])
                            # self.logger.info(index)
                            if index in self.lables:
                                shutil.copy(source, target)
                                self.logger.info(f"copy source={source} target={target}")
                                source = source.replace('.txt', '.jpg')
                                target = self.args.target.replace('.txt', '.jpg')
                                shutil.copy(source, target)
                                self.logger.info(f"copy source={source} target={target}")
                                self.count += 1
                                break
                else:
                    shutil.copy(source, target)
                    self.logger.info(f"copy source={file} target={target}")
                try:
                    shutil.copy(image, target.replace('.txt', '.jpg', 1))
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
        pass

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

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("change")

        # self.indexs = {}
        # self.classes = []
        self.editable = {}
        self.total = {}

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)

        return (files)

    def input(self):
        try:

            self.logger.info(f"search={self.args.search}")
            self.logger.info(f"replace={self.args.replace}")

            for n in range(0, len(self.args.search)):
                self.editable[self.args.search[n]] = self.args.replace[n]

            self.logger.info(f"editable={self.editable}")

            self.files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)
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
                    if filename.lower() == 'classes.txt':
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
                                        self.logger.info(f"search={s} replace={r} line={line.strip()}")
                                        break

                                # if index > len(self.classes):
                                if index not in self.total:
                                    self.total[index] = 0
                                self.total[index] += 1

                                lines.append(line)
                        if len(lines) > 0:
                            with open(file, "w", encoding="utf-8") as newfile:
                                # txts = [line + "\n" for line in lines]
                                # newfile.writelines(txts)
                                # newfile.write(line)
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
        pass

    def main(self):
        if self.args.source and self.args.search and self.args.replace:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()


class YoloLabel(Common):
    count = 0

    def __init__(self, parser, args):

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("label")

        self.indexs = {}

    def classes(self):
        classes = os.path.join(self.args.source, 'classes.txt')
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error(f"classes.txt 文件不存在！")
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
        self.files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)
        self.logger.info(f"files total={len(self.files)}")
        # progress = {}
        with tqdm(total=len(self.files), ncols=150) as progress:
            for file in self.files:
                progress.set_description(file)
                filename = os.path.basename(file)
                self.logger.info(f"file={file}")
                try:
                    if filename.lower() == 'classes.txt':
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
            classes = os.path.join(self.args.source, 'classes.txt')
            if not os.path.isfile(classes):
                print(f"classes.txt 文件不存在: {classes}")
                self.logger.error(f"classes.txt 文件不存在！")
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

        self.files = glob.glob(f'{self.args.source}/**/*.txt', recursive=True)
        self.logger.info(f"files total={len(self.files)}")
        data = {}

        with tqdm(total=len(self.files), ncols=100) as progress:

            for file in self.files:
                # progress.set_description(file)
                filename = os.path.basename(file)
                self.logger.info(f"file={file}")
                try:
                    if filename.lower() == 'classes.txt':
                        self.logger.info(f"skip file={file}")
                        continue
                    else:
                        with open(file, "r", encoding="utf-8") as original:
                            for line in original.readlines():
                                index = line.strip().split(" ")[0]
                                if index not in data.keys():
                                    data[index] = []
                                if index in self.args.search:
                                    # print(file)
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
        elif self.args.source and self.args.search:
            self.search()
        else:
            self.parser.print_help()
            exit()


class YoloResize(Common):
    total = {"未处理": 0, "已处理": 0}

    def __init__(self, parser, args):

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("resize")

    def resize(self, image):
        # from PIL import Image
        # 加载图像
        # image = Image.open('path_to_your_image.jpg')
        # 计算缩放因子
        width, height = image.size
        # print(width, height)
        if max(width, height) > self.args.imgsz:
            if width > height:
                ratio = width / self.args.imgsz
                width = self.args.imgsz
                height = int(height / ratio)
            else:
                ratio = height / self.args.imgsz
                width = int(width / ratio)
                height = self.args.imgsz

            # print(ratio)
            # print(width, height)
            image = image.resize((width, height))
            image = ImageOps.exif_transpose(image)
            return image
        return image

    def images(self, source, target):
        try:

            original = Image.open(source)
            width, height = original.size
            # print(target)
            if max(width, height) < self.args.imgsz:
                shutil.copyfile(source, target)
                self.total['未处理'] += 1
                self.logger.info(f"skip source={source} target={target}")
            else:
                image = self.resize(original)
                image.save(target)
                self.total['已处理'] += 1
                self.logger.info(f"size={original.size} resize={image.size} source={source} target={target}")

        except Exception as e:
            # log.error(e)
            print("images: ", e)
            exit()
        pass

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.target):
                    shutil.rmtree(self.args.target)
                if self.args.output and os.path.exists(self.args.output):
                    shutil.rmtree(self.args.output)
                    os.makedirs(os.path.join(self.args.output), exist_ok=True)

            os.makedirs(os.path.join(self.args.target), exist_ok=True)

        except Exception as e:
            # log.error(e)
            print("input: ", repr(e))
            exit()

        self.files = glob.glob(f'{self.args.source}/**/*.jpg', recursive=True)
        self.logger.info(f"files total={len(self.files)}")

        # print(self.files)
        # self.model = YOLO(f"{self.basedir}/model/Tongue/weights/best.pt")

    def process(self):
        with tqdm(total=len(self.files), ncols=120) as progress:
            for source in self.files:
                progress.set_description(source)

                # target = source.replace(self.args.source, self.args.target)
                target = source.replace(os.path.join(self.args.source), os.path.join(self.args.target))
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                self.images(source, target)
                progress.update(1)

    def output(self):
        tables = [["事件", "统计"]]
        for key, value in self.total.items():
            tables.append([key, value])
        tables.append(["合计", len(self.files)])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())
        pass

    def main(self):

        if self.args.source and self.args.target:
            if self.args.source == self.args.target:
                print("目标文件夹不能与原始图片文件夹相同")
                exit()
            self.input()
            self.process()
            self.output()

        else:
            self.parser.print_help()
            exit()


class ImageCrop(Common):
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)
    expand = 50

    # border = 10
    total = {"未处理": 0, "已处理": 0}

    def __init__(self, parser, args):
        self.files = []
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("crop")

    def border(self, original, xyxy):
        # original = Image.open(source)
        width, height = original.size
        x0, y0, x1, y1 = map(int, xyxy)

        if x0 - self.expand < 0:
            x0 = 0
        else:
            x0 -= self.expand

        if y0 - self.expand < 0:
            y0 = 0
        else:
            y0 -= self.expand

        if x1 + self.expand > width:
            x1 = width
        else:
            x1 += self.expand

        if y1 + self.expand > height:
            y1 = height
        else:
            y1 += self.expand

        # print(f"xyxy={xyxy}")
        # print(x0, y0, x1, y1)
        # crop = tuple(map(int, xyxy))
        crop = tuple((x0, y0, x1, y1))
        tongue = original.crop(crop)
        # tongue = self.resize(tongue)
        # crop.save(output)
        width, height = tongue.size
        # width += self.border
        # height += self.border
        image = Image.new('RGB', (width, height), self.background)
        image.paste(tongue, (
            int(width / 2) - int(tongue.size[0] / 2), int(height / 2) - int(tongue.size[1] / 2)))
        return image

    def crop(self, source: str, target: str):
        if not os.path.exists(source):
            return None

        try:

            image = cv2.imread(source)
            if image is None:
                return None

            results = self.model(source, verbose=False)

            for result in results:
                # 提取检测到的每个目标的边界框
                boxes = result.boxes.data.cpu().numpy()  # YOLO 边界框格式：[x1, y1, x2, y2, confidence, class]
                # print(result.boxes.data.tolist())
                if self.args.output:
                    result.save(filename=os.path.join(self.args.output, os.path.basename(source)))
                    result.save_crop(save_dir=os.path.join(self.args.output, 'crop'), file_name="detection")
                # print(boxes)

                # x1, y1, x2, y2, conf, cls = map(int, box[0][:6])
                # cropped = image[y1:y2, x1:x2]
                # # output = os.path.join(self.args.target, os.path.basename(source))
                # cv2.imwrite(target, cropped)

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    cropped = image[y1:y2, x1:x2]
                    output = os.path.join(self.args.target,
                                          f"{os.path.splitext(os.path.basename(source))[0]}_{idx}.jpg")
                    cv2.imwrite(target, cropped)
                    self.total['已处理'] += 1
                    self.logger.info(f"Saved cropped image: {target}")
                    return target
            self.total['未处理'] += 1
        except Exception as e:
            print(e)
            self.logger.error(e)
            exit()
        return None

    def input(self):
        try:
            if self.args.clean:
                if os.path.exists(self.args.target):
                    shutil.rmtree(self.args.target)
                if self.args.output and os.path.exists(self.args.output):
                    shutil.rmtree(self.args.output)
                    os.makedirs(os.path.join(self.args.output), exist_ok=True)

            os.makedirs(os.path.join(self.args.target), exist_ok=True)

        except Exception as e:
            self.logger.error(e)
            print("input: ", repr(e))
            exit()

        self.files = glob.glob(f'{self.args.source}/**/*.jpg', recursive=True)
        self.logger.info(f"files total={len(self.files)}")

        # print(self.files)
        self.model = YOLO(self.args.model)
        self.logger.info(f"loading model={self.args.model}")

    def process(self):
        with tqdm(total=len(self.files), ncols=120) as progress:
            for source in self.files:

                source = os.path.join(source)
                target = source.replace(os.path.join(self.args.source), os.path.join(self.args.target))

                progress.set_description(source)

                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                self.crop(source, target)
                self.logger.info(f"images source={source} target={target}")
                progress.update(1)

    def output(self):
        tables = [["事件", "统计"]]
        for key, value in self.total.items():
            tables.append([key, value])
        tables.append(["合计", len(self.files)])
        table = Texttable(max_width=100)
        table.add_rows(tables)
        print(table.draw())
        pass

    def main(self):

        if self.args.source and self.args.target and self.args.model:
            if self.args.source == self.args.target:
                print("目标文件夹不能与原始图片文件夹相同")
                exit()
            self.input()
            self.process()
            self.output()

        else:
            self.parser.print_help()
            exit()


class Classify(Common):
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    checklists = []
    dataset = {}
    crop = False
    model = None

    def __init__(self, parser, args):

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("classify")

        self.basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(self.basedir)
        # print(basedir)
        # exit()

    def boxes(self, source: str, target: str) -> None:
        if not os.path.exists(source):
            return None
        if not self.model:
            return None
        results = self.model(source, verbose=self.args.verbose)
        image = cv2.imread(source)
        filename, extension = os.path.splitext(os.path.basename(target))
        for result in results:
            # print(result)

            if self.args.output:
                result.save(filename=os.path.join(self.args.output, os.path.basename(source)))
            try:
                boxes = result.boxes.data.cpu().numpy()  # YOLO 边界框格式：[x1, y1, x2, y2, confidence, class]
                #     # print(result.boxes.data.tolist())
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    cropped = image[y1:y2, x1:x2]
                    output = os.path.join(os.path.dirname(target), f"{filename}_{idx}{extension}")
                    cv2.imwrite(output, cropped)
                    # print(f"Saved cropped image: {output}")
                if len(boxes) > 1:
                    self.checklists.append(target)
                    if self.args.checklist:
                        result.save_crop(save_dir=os.path.join(self.args.checklist, 'crop'), file_name=filename)
                        result.save(filename=os.path.join(self.args.checklist, os.path.basename(source)))
                    # print(boxes)
            except Exception as e:
                # log.error(e)
                print("boxes: ", e)
                exit()

    def source(self, label, filename):
        return os.path.join(self.args.source, label, filename)

    def target(self, mode, label, filename):
        if self.args.uuid:
            extension = os.path.splitext(filename)[1]
            path = os.path.join(self.args.target, f"{mode}", label, f"{uuid.uuid4()}{extension}")
        else:
            path = os.path.join(self.args.target, f"{mode}", label, filename)
        return path

    def train(self):

        for label, files in self.dataset.items():
            with tqdm(total=len(files), ncols=100) as progress:
                progress.set_description(f"train/{label}")
                for name in files:
                    try:
                        source = self.source(label, name)
                        # print(input)
                        target = self.target('train', label, name)
                        # print(target)
                        if self.crop:
                            # self.crop(source,target)
                            self.boxes(source, target)
                        else:
                            # print(f"COPY train source={source}, target={target}")
                            shutil.copyfile(source, target)
                    except Exception as e:
                        # log.error(e)
                        print("train: ", e)
                        exit()
                    progress.update(1)

        for cls in self.scandir(os.path.join(self.args.target, 'train')):
            self.dataset[cls] = self.scanfile(os.path.join(self.args.target, 'train', cls))
        # print(self.dataset)

    def test(self):

        for cls, files in self.dataset.items():
            if len(files) < self.args.test:
                self.args.test = len(files)
            tests = random.sample(files, self.args.test)
            with tqdm(total=len(tests), ncols=100) as progress:
                progress.set_description(f"test/{cls}")
                for image in tests:
                    try:
                        source = os.path.join(self.args.target, 'train', cls, image)
                        target = self.target('test', cls, image)
                        # print(f"source={source} target={target}")
                        shutil.copyfile(source, target)
                    except Exception as e:
                        # log.error(e)
                        print("test: ", e)
                        exit()
                    progress.update(1)

    def val(self):

        for cls, files in self.dataset.items():
            if len(files) < self.args.test:
                self.args.test = len(files)
            vals = random.sample(files, self.args.test)
            with tqdm(total=len(vals), ncols=100) as progress:
                progress.set_description(f"val/{cls}")
                for image in vals:
                    try:
                        source = os.path.join(self.args.target, 'train', cls, image)
                        # print(input)
                        target = self.target('val', cls, image)
                        shutil.copyfile(source, target)
                    except Exception as e:
                        # log.error(e)
                        print("test: ", e)
                        exit()
                    progress.update(1)

    def input(self):

        if self.args.clean:
            if os.path.exists(self.args.target):
                shutil.rmtree(self.args.target)
            if self.args.output and os.path.exists(self.args.output):
                shutil.rmtree(self.args.output)
            if self.args.checklist and os.path.exists(self.args.checklist):
                shutil.rmtree(self.args.checklist)

        self.mkdirs(os.path.join(self.args.target))
        if self.args.output:
            self.mkdirs(os.path.join(self.args.output))
        if self.args.checklist:
            self.mkdirs(os.path.join(self.args.checklist))

        directory = [
            'train', 'test', 'val'
        ]

        for cls in self.scandir(os.path.join(self.args.source)):
            self.dataset[cls] = self.scanfile(os.path.join(self.args.source, cls))
            for dir in directory:
                self.mkdirs(os.path.join(self.args.target, dir, cls))
        # print(self.dataset)
        if self.args.crop:
            if self.args.model and os.path.isfile(self.args.model):
                self.model = YOLO(self.args.model)
                self.crop = True
            else:
                print(f"载入模型失败 {self.args.model}")
                exit()

        pass

    def process(self):
        self.train()
        self.test()
        self.val()
        pass

    def output(self):
        if self.checklists:
            tables = [["检查列表"]]
            for file in self.checklists:
                tables.append([file])
            table = Texttable(max_width=100)
            table.add_rows(tables)
            print(table.draw())
        pass

    def main(self):
        if self.args.source and self.args.target:
            self.input()
            self.process()
            self.output()
        else:
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

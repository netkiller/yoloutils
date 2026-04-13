import logging
import os
import random
import shutil
import sys
import uuid

import cv2
from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class YoloClassify:
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    checklists = []
    dataset = {}
    crop = False
    model = None

    def __init__(self, parser, args):
        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        parser.add_argument(
            "--output", type=str, default=None, help="输出识别图像", metavar=""
        )
        parser.add_argument(
            "--checklist", type=str, default=None, help="输出识别图像", metavar=""
        )
        parser.add_argument(
            "--test", type=int, default=10, help="测试数量", metavar=100
        )
        # self.classify.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        parser.add_argument(
            "--crop", action="store_true", default=False, help="裁剪"
        )
        parser.add_argument(
            "--model", type=str, default=None, help="裁剪模型", metavar=""
        )
        parser.add_argument(
            "--uuid", action="store_true", default=False, help="重命名图片为UUID"
        )
        parser.add_argument(
            "--verbose", action="store_true", default=False, help="过程输出"
        )
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.basedir = BASE_DIR
        sys.path.append(self.basedir)

        self.checklists = []
        self.dataset = {}
        self.crop = False
        self.model = None

    def mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def scanfile(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isfile(os.path.join(path, name)):
                files.append(name)
        return files

    def scandir(self, path):
        files = []
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                files.append(name)
        return files

    def boxes(self, source: str, target: str) -> None:
        if not os.path.exists(source):
            return None
        if not self.model:
            return None
        results = self.model(source, verbose=self.args.verbose)
        image = cv2.imread(source)
        filename, extension = os.path.splitext(os.path.basename(target))
        for result in results:
            if self.args.output:
                result.save(filename=os.path.join(self.args.output, os.path.basename(source)))
            try:
                boxes = result.boxes.data.cpu().numpy()
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    cropped = image[y1:y2, x1:x2]
                    output = os.path.join(os.path.dirname(target), f"{filename}_{idx}{extension}")
                    cv2.imwrite(output, cropped)
                if len(boxes) > 1:
                    self.checklists.append(target)
                    if self.args.checklist:
                        result.save_crop(
                            save_dir=os.path.join(self.args.checklist, "crop"),
                            file_name=filename,
                        )
                        result.save(
                            filename=os.path.join(
                                self.args.checklist, os.path.basename(source)
                            )
                        )
            except Exception as e:
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
                        target = self.target("train", label, name)
                        if self.crop:
                            self.boxes(source, target)
                        else:
                            shutil.copyfile(source, target)
                    except Exception as e:
                        print("train: ", e)
                        exit()
                    progress.update(1)

        for cls in self.scandir(os.path.join(self.args.target, "train")):
            self.dataset[cls] = self.scanfile(os.path.join(self.args.target, "train", cls))

    def test(self):
        for cls, files in self.dataset.items():
            if len(files) < self.args.test:
                self.args.test = len(files)
            tests = random.sample(files, self.args.test)
            with tqdm(total=len(tests), ncols=100) as progress:
                progress.set_description(f"test/{cls}")
                for image in tests:
                    try:
                        source = os.path.join(self.args.target, "train", cls, image)
                        target = self.target("test", cls, image)
                        shutil.copyfile(source, target)
                    except Exception as e:
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
                        source = os.path.join(self.args.target, "train", cls, image)
                        target = self.target("val", cls, image)
                        shutil.copyfile(source, target)
                    except Exception as e:
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

        directory = ["train", "test", "val"]
        for cls in self.scandir(os.path.join(self.args.source)):
            self.dataset[cls] = self.scanfile(os.path.join(self.args.source, cls))
            for dir_name in directory:
                self.mkdirs(os.path.join(self.args.target, dir_name, cls))

        if self.args.crop:
            if self.args.model and os.path.isfile(self.args.model):
                self.model = YOLO(self.args.model)
                self.crop = True
            else:
                print(f"载入模型失败 {self.args.model}")
                exit()

    def process(self):
        self.train()
        self.test()
        self.val()

    def output(self):
        if self.checklists:
            tables = [["检查列表"]]
            for file in self.checklists:
                tables.append([file])
            table = Texttable(max_width=100)
            table.add_rows(tables)
            print(table.draw())

    def main(self):
        if self.args.source and self.args.target:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()

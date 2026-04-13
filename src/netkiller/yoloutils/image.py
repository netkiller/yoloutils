import glob
import logging
import os
import shutil

import cv2
from PIL import Image, ImageOps
from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO


class YoloImageCrop:
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)
    expand = 50

    # border = 10
    total = {"未处理": 0, "已处理": 0}

    def __init__(self, parser, args):
        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        parser.add_argument(
            "--model", type=str, default=None, metavar="best.pt", help="模型"
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Yolo 输出目录",
            metavar="/tmp/output",
        )
        self.files = []
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger("crop")
        self.total = {"未处理": 0, "已处理": 0}

    def border(self, original, xyxy):
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

        crop = tuple((x0, y0, x1, y1))
        tongue = original.crop(crop)
        width, height = tongue.size
        image = Image.new("RGB", (width, height), self.background)
        image.paste(
            tongue,
            (
                int(width / 2) - int(tongue.size[0] / 2),
                int(height / 2) - int(tongue.size[1] / 2),
            ),
        )
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
                boxes = result.boxes.data.cpu().numpy()
                if self.args.output:
                    result.save(
                        filename=os.path.join(
                            self.args.output, os.path.basename(source)
                        )
                    )
                    result.save_crop(
                        save_dir=os.path.join(self.args.output, "crop"),
                        file_name="detection",
                    )

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    cropped = image[y1:y2, x1:x2]
                    output = os.path.join(
                        self.args.target,
                        f"{os.path.splitext(os.path.basename(source))[0]}_{idx}.jpg",
                    )
                    cv2.imwrite(target, cropped)
                    self.total["已处理"] += 1
                    self.logger.info(f"Saved cropped image: {target}")
                    return target
            self.total["未处理"] += 1
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

        self.files = glob.glob(f"{self.args.source}/**/*.jpg", recursive=True)
        self.logger.info(f"files total={len(self.files)}")

        self.model = YOLO(self.args.model)
        self.logger.info(f"loading model={self.args.model}")

    def process(self):
        with tqdm(total=len(self.files), ncols=120) as progress:
            for source in self.files:
                source = os.path.join(source)
                target = source.replace(
                    os.path.join(self.args.source), os.path.join(self.args.target)
                )

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


class YoloImageResize:
    total = {"未处理": 0, "已处理": 0}

    def __init__(self, parser, args):
        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )
        parser.add_argument(
            "--imgsz", type=int, default=640, help="长边尺寸", metavar=640
        )
        parser.add_argument(
            "--output", type=str, default=None, help="输出识别图像", metavar=""
        )
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)
        self.total = {"未处理": 0, "已处理": 0}

    def resize(self, image):
        width, height = image.size
        if max(width, height) > self.args.imgsz:
            if width > height:
                ratio = width / self.args.imgsz
                width = self.args.imgsz
                height = int(height / ratio)
            else:
                ratio = height / self.args.imgsz
                width = int(width / ratio)
                height = self.args.imgsz

            image = image.resize((width, height))
            image = ImageOps.exif_transpose(image)
            return image
        return image

    def images(self, source, target):
        try:
            original = Image.open(source)
            width, height = original.size
            if max(width, height) < self.args.imgsz:
                shutil.copyfile(source, target)
                self.total["未处理"] += 1
                self.logger.info(f"skip source={source} target={target}")
            else:
                image = self.resize(original)
                image.save(target)
                self.total["已处理"] += 1
                self.logger.info(
                    f"size={original.size} resize={image.size} source={source} target={target}"
                )
        except Exception as e:
            print("images: ", e)
            exit()

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
            print("input: ", repr(e))
            exit()

        self.files = glob.glob(f"{self.args.source}/**/*.jpg", recursive=True)
        self.logger.info(f"files total={len(self.files)}")

    def process(self):
        with tqdm(total=len(self.files), ncols=120) as progress:
            for source in self.files:
                progress.set_description(source)

                target = source.replace(
                    os.path.join(self.args.source), os.path.join(self.args.target)
                )
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

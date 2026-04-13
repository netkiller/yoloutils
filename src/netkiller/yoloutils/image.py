import glob
import logging
import os
import shutil

import cv2
from PIL import Image, ImageOps
from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO

try:
    from . import Common
except ImportError:
    from __init__ import Common


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

        modelgroup = parser.add_argument_group(title="基于模型裁切", description="用指定模型识别后，将 box 框内的图像保存指定目录")
        modelgroup.add_argument(
            "--model", type=str, default=None, metavar="best.pt", help="模型"
        )
        modelgroup.add_argument(
            "--output",
            type=str,
            default=None,
            help="Yolo 输出目录",
            metavar="/tmp/output",
        )
        txt = parser.add_argument_group(title="基于txt标准裁切", description="用.txt文件中的box框为基准，向外扩展裁切")
        txt.add_argument('--txt', action="store_true", default=False, help='自动标注')
        txt.add_argument('--imgsz', type=int, default=640, help='矩形长边', metavar=640)

        self.files = []
        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)
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

    def model(self, source: str, target: str):
        if not os.path.exists(source):
            return None

        try:
            image = cv2.imread(source)
            if image is None:
                return None

            results = self.yolo(source, verbose=False)

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

    def txt(self, source: str, target: str, imgsz: int = 640):
        label_source = f"{os.path.splitext(source)[0]}.txt"
        label_target = f"{os.path.splitext(target)[0]}.txt"

        if not os.path.exists(label_source):
            self.total["未处理"] += 1
            self.logger.warning(f"txt missing source={source} label={label_source}")
            return None

        image = cv2.imread(source)
        if image is None:
            self.total["未处理"] += 1
            self.logger.warning(f"image invalid source={source}")
            return None

        height, width = image.shape[:2]
        if imgsz <= 0:
            self.total["未处理"] += 1
            self.logger.warning(f"invalid imgsz={imgsz} source={source}")
            return None

        with open(label_source, "r", encoding="utf-8") as file:
            rows = [line.strip() for line in file if line.strip()]

        if not rows:
            self.total["未处理"] += 1
            self.logger.warning(f"txt empty source={source} label={label_source}")
            return None

        parsed = []
        for row in rows:
            parts = row.split()
            if len(parts) < 5:
                continue
            try:
                cls = parts[0]
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            parsed.append((cls, cx, cy, bw, bh))

        if not parsed:
            self.total["未处理"] += 1
            self.logger.warning(f"txt parse failed source={source} label={label_source}")
            return None

        first_cls, first_cx, first_cy, first_bw, first_bh = parsed[0]
        box_w_px = first_bw * width
        box_h_px = first_bh * height

        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.makedirs(os.path.dirname(label_target), exist_ok=True)

        # 框尺寸超过 imgsz，按原图/原标签复制，不做裁切
        if box_w_px > imgsz or box_h_px > imgsz:
            shutil.copy2(source, target)
            shutil.copy2(label_source, label_target)
            self.total["已处理"] += 1
            self.logger.info(
                f"txt crop skipped(box>{imgsz}) source={source} target={target}"
            )
            return target

        center_x = first_cx * width
        center_y = first_cy * height
        half = imgsz / 2.0
        pad = int(half)

        padded = cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=self.background,
        )

        crop_left = center_x - half
        crop_top = center_y - half
        crop_right = crop_left + imgsz
        crop_bottom = crop_top + imgsz

        left = int(round(crop_left + pad))
        top = int(round(crop_top + pad))
        right = left + imgsz
        bottom = top + imgsz

        cropped = padded[top:bottom, left:right]
        if cropped.shape[0] != imgsz or cropped.shape[1] != imgsz:
            self.total["未处理"] += 1
            self.logger.warning(
                f"txt crop invalid shape={cropped.shape[:2]} imgsz={imgsz} source={source}"
            )
            return None

        adjusted_rows = []
        for cls, cx, cy, bw, bh in parsed:
            bx = cx * width
            by = cy * height
            bw_px = bw * width
            bh_px = bh * height
            x1 = bx - bw_px / 2.0
            y1 = by - bh_px / 2.0
            x2 = bx + bw_px / 2.0
            y2 = by + bh_px / 2.0

            ix1 = max(x1, crop_left)
            iy1 = max(y1, crop_top)
            ix2 = min(x2, crop_right)
            iy2 = min(y2, crop_bottom)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            new_cx = ((ix1 + ix2) / 2.0 - crop_left) / imgsz
            new_cy = ((iy1 + iy2) / 2.0 - crop_top) / imgsz
            new_bw = (ix2 - ix1) / imgsz
            new_bh = (iy2 - iy1) / imgsz
            adjusted_rows.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

        if not adjusted_rows:
            self.total["未处理"] += 1
            self.logger.warning(f"txt crop no labels after clip source={source}")
            return None

        cv2.imwrite(target, cropped)
        with open(label_target, "w", encoding="utf-8") as file:
            file.write("\n".join(adjusted_rows) + "\n")

        self.total["已处理"] += 1
        self.logger.info(f"txt crop saved source={source} target={target}")
        return target

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

        files = glob.glob(f"{self.args.source}/**/*", recursive=True)
        self.files = [
            f for f in files if os.path.isfile(f) and f.lower().endswith(Common.image_exts)
        ]
        self.logger.info(f"files total={len(self.files)}")

        if self.args.model:
            self.yolo = YOLO(self.args.model)
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
                if self.args.model:
                    self.model(source, target)
                    self.logger.info(f"images crop=model source={source} target={target}")
                elif self.args.txt:
                    self.txt(source, target, imgsz=self.args.imgsz)
                    self.logger.info(f"images crop=txt source={source} target={target}")
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
        if self.args.source and self.args.target and (self.args.model or self.args.txt):
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

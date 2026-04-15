import csv
import glob
import logging
import os
import random
import shutil
import uuid

import cv2
import yaml
from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO

try:
    from . import BASE_DIR, Common
except ImportError:
    # Support direct script execution (python labelimg.py ...)
    from __init__ import BASE_DIR, Common


class YoloLabelimg(Common):
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)

    def __init__(self):
        self.basedir = BASE_DIR

        self.classes = []
        self.lables = {}
        self.missed = []
        self.files = {}
        self.logger = logging.getLogger(__class__.__name__)

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

        files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)
        with tqdm(total=len(files), ncols=120) as progress:
            progress.set_description("file scanning")
            for source in files:
                progress.set_postfix_str(f"file={os.path.basename(source)[:36]:<36}")
                if source.endswith("classes.txt"):
                    progress.update(1)
                    continue
                if os.path.getsize(source) == 0:
                    self.missed.append(source)
                    self.logger.warning(f"标注文件为空: {source}")
                    progress.update(1)
                    continue
                for ext in Common.image_exts:
                    if os.path.exists(f"{os.path.splitext(source)[0]}{ext}"):
                        self.files[source] = f"{os.path.splitext(source)[0]}{ext}"
                        break
                else:
                    self.missed.append(source)
                    self.logger.warning(f"标注文件缺少配对图片: {source}")
                progress.update(1)

        with tqdm(total=len(directory), ncols=120) as progress:
            progress.set_description(f"yolo init")
            for dir in directory:
                progress.set_postfix_str(f"dir={dir[:36]:<36}")
                self.mkdirs(os.path.join(self.args.target, dir))
                progress.update(1)

    def process(self):
        # images =  glob.glob('*.jpg', root_dir=self.args.source)
        # labels = glob.glob('*.txt', root_dir=self.args.source)
        with (
            tqdm(
                total=len(self.files),
                ncols=150,
                bar_format="{desc} {percentage:3.0f}%|{bar:58}| {n:>4.0f}/{total:>4.0f} {postfix}",
            ) as images,
            tqdm(
                total=len(self.files),
                ncols=150,
                bar_format="{desc} {percentage:3.0f}%|{bar:58}| {n:>4.0f}/{total:>4.0f} {postfix}",
            ) as train,
        ):
            for source in self.files.keys():

                train.set_description("train/labels")
                train.set_postfix_str(f"file={os.path.basename(source)[:36]:<36}")

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
                # 图片复制
                images.set_description("train/images")
                image = self.files[source]
                images.set_postfix_str(f"file={os.path.basename(image)[:36]:<36}")

                if self.args.uuid:
                    target = os.path.join(
                        self.args.target,
                        "train/images",
                        f"{name}{os.path.splitext(image)[1]}",
                    )
                else:
                    target = os.path.join(
                        self.args.target, "train/images", os.path.basename(image)
                    )
                shutil.copy(image, target)
                self.logger.info(
                    f"train/images source={image} target={target} name={name}"
                )
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

                        for ext in Common.image_exts:
                            source = os.path.join(
                                self.args.target, "train/images", f"{name}{ext}"
                            )
                            if os.path.exists(source):
                                target = os.path.join(
                                    self.args.target, "val/images", f"{name}{ext}"
                                )
                                shutil.copy(source, target)
                                self.logger.info(
                                    f"val/images copy label={label} source={source} target={target}"
                                )
                                break
                        else:
                            self.logger.warning(
                                f"val/images missing train image label={label} name={name}"
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

        tables = [["丢失图像"]]
        if self.missed:
            for file in self.missed:
                tables.append([os.path.relpath(file, self.args.source)])
        else:
            tables.append(["（无）"])
        table = Texttable(max_width=160)
        table.add_rows(tables)
        print(table.draw())
        print(f"Total: {len(self.files) + len(self.missed)}, Lost: {len(self.missed)}")

        for file in self.missed:
            self.logger.warning(f"丢失文件 {file}")

        tables = [["标签", "数量"]]
        for label, files in self.lables.items():
            # if len(files) == 0:
            #     continue
            tables.append([label, len(files)])
        table = Texttable(max_width=160)
        table.add_rows(tables)
        print(table.draw())

    def main(self, **args):
        self.args = args
        self.logger.info("Start")
        self.input()
        self.process()
        self.output()
        self.logger.info("Done")


class YoloLabelimgAutomatic(Common):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def __init__(self, parser, args):
        self.basedir = BASE_DIR

        parser.add_argument("--source", type=str, default=None, help="图片来源地址")
        parser.add_argument("--target", type=str, default=None, help="图片目标地址")
        parser.add_argument(
            "--clean", action="store_true", default=False, help="清理之前的数据"
        )

        parser.add_argument('--model', type=str, default=None, help='载入模型', metavar="best.pt")
        parser.add_argument('--conf', type=float, default=None, help='置信度阈值', metavar=0.5)
        parser.add_argument('--csv', default=None, type=str, help='报告输出，哪些文件已经标准，哪些没有标注', metavar="report.csv")
        parser.add_argument('--output', type=str, default=None, help='输出标注效果', metavar="/path/to/output")

        self.parser = parser
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)
        self.model = None
        self.files = []
        self.unlabeled_files = []
        self.auto_stats = {
            "图片总数": 0,
            "已标注": 0,
            "未标注": 0,
            "标注框总数": 0
        }

    def _clean_paths(self):
        paths = []
        for path in (self.args.target, self.args.output):
            if path and path not in paths:
                paths.append(path)
        return paths

    def _confirm_clean(self, paths):
        if not paths:
            return True

        print("检测到 --clean，将删除以下目录：")
        for path in paths:
            print(f"- {path}")
        try:
            answer = input("是否继续？[y/N]: ").strip().lower()
        except EOFError:
            answer = ""

        if answer in ("y", "yes"):
            return True

        print("已取消清理操作")
        self.logger.info("cancel clean operation by user")
        return False

    def input(self):
        if not os.path.isdir(self.args.source):
            print(f"source 目录不存在: {self.args.source}")
            self.logger.error(f"source 目录不存在: {self.args.source}")
            exit()
        if self.args.conf is not None and not (0.0 <= self.args.conf <= 1.0):
            print(f"--conf 超出范围: {self.args.conf}，必须在 0~1 之间")
            self.logger.error(f"--conf out of range: {self.args.conf}")
            exit()

        try:
            if self.args.clean:
                clean_paths = self._clean_paths()
                if not self._confirm_clean(clean_paths):
                    exit()
                for path in clean_paths:
                    if os.path.exists(path):
                        shutil.rmtree(path)
            os.makedirs(self.args.target, exist_ok=True)
            if self.args.output:
                os.makedirs(self.args.output, exist_ok=True)
        except Exception as e:
            self.logger.error(f"auto input: {repr(e)}")
            print("auto input: ", repr(e))
            exit()

        try:
            self.model = YOLO(self.args.model)
        except FileNotFoundError as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e, f" {self.args.model}")
            exit()
        except Exception as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e)
            exit()

        files = glob.glob(f"{self.args.source}/**/*", recursive=True)
        self.files = sorted(
            [
                f
                for f in files
                if os.path.isfile(f) and os.path.splitext(f)[1].lower() in self.image_exts
            ]
        )
        self.auto_stats["图片总数"] = len(self.files)
        self.logger.info(f"auto files total={len(self.files)}")

    def _classes_from_model(self):
        names = self.model.names
        if isinstance(names, (list, tuple)):
            return [str(name) for name in names]
        if isinstance(names, dict):
            result = []

            def sort_key(value):
                try:
                    return (0, int(value))
                except (TypeError, ValueError):
                    return (1, str(value))

            for key in sorted(names.keys(), key=sort_key):
                result.append(str(names.get(key)))
            return result
        return []

    def _save_visualized_image(self, source, relpath, results):
        if not self.args.output:
            return

        output_image = os.path.join(self.args.output, relpath)
        os.makedirs(os.path.dirname(output_image), exist_ok=True)

        try:
            if results:
                plotted = results[0].plot()
                if plotted is not None and cv2.imwrite(output_image, plotted):
                    return
            shutil.copy2(source, output_image)
        except Exception as e:
            self.logger.error(
                f"save visualized image failed source={source} target={output_image} err={repr(e)}"
            )

    def process(self):
        with tqdm(total=len(self.files), ncols=140) as progress:
            for source in self.files:
                progress.set_description(source)
                relpath = os.path.relpath(source, self.args.source)
                target_image = os.path.join(self.args.target, relpath)
                target_label = f"{os.path.splitext(target_image)[0]}.txt"

                os.makedirs(os.path.dirname(target_image), exist_ok=True)
                shutil.copy2(source, target_image)

                lines = []
                results = []
                try:
                    predict_kwargs = {"verbose": False}
                    if self.args.conf is not None:
                        predict_kwargs["conf"] = self.args.conf
                    results = self.model.predict(source, **predict_kwargs)
                    for result in results:
                        boxes = result.boxes
                        if boxes is None or boxes.cls is None or boxes.xywhn is None:
                            continue

                        classes = boxes.cls.cpu().tolist()
                        coords = boxes.xywhn.cpu().tolist()
                        confs = (
                            boxes.conf.cpu().tolist()
                            if boxes.conf is not None
                            else [None] * len(coords)
                        )
                        for idx, xywh in enumerate(coords):
                            if self.args.conf is not None:
                                conf_value = confs[idx] if idx < len(confs) else None
                                if conf_value is None or conf_value <= self.args.conf:
                                    continue
                            cls_id = int(classes[idx])
                            lines.append(
                                f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}"
                            )
                except Exception as e:
                    self.logger.error(f"auto process {source}: {repr(e)}")
                self._save_visualized_image(source, relpath, results)

                with open(target_label, "w", encoding="utf-8") as file:
                    if lines:
                        file.write("\n".join(lines) + "\n")

                if lines:
                    self.auto_stats["已标注"] += 1
                    self.auto_stats["标注框总数"] += len(lines)
                else:
                    self.unlabeled_files.append(relpath)
                progress.update(1)

    def _print_summary(self):
        self.auto_stats["未标注"] = len(self.unlabeled_files)
        print(
            "统计结果 "
            f"图片总数：{self.auto_stats['图片总数']}，"
            f"已标注：{self.auto_stats['已标注']}，"
            f"未标注：{self.auto_stats['未标注']}，"
            f"标注框总数：{self.auto_stats['标注框总数']}"
        )

    def _print_unlabeled_files(self):
        tables = [["未标注文件名"]]
        unlabeled = sorted(self.unlabeled_files)
        if unlabeled:
            for filename in unlabeled:
                tables.append([filename])
        else:
            tables.append(["（无）"])
        table = Texttable(max_width=160)
        table.add_rows(tables)
        print(table.draw())

    def _write_report_csv(self):
        if not self.args.csv:
            return

        report_path = self.args.csv
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["未标注文件名"])
            unlabeled = sorted(self.unlabeled_files)
            if unlabeled:
                for filename in unlabeled:
                    writer.writerow([filename])
            else:
                writer.writerow(["（无）"])

        self.logger.info(f"report csv: {report_path}")
        print(f"report csv: {report_path}")

    def output(self):
        classes = self._classes_from_model()
        with open(
                os.path.join(self.args.target, "classes.txt"), "w", encoding="utf-8"
        ) as file:
            if classes:
                file.write("\n".join(classes) + "\n")

        self._print_unlabeled_files()
        self._print_summary()
        self._write_report_csv()

    def main(self):
        if self.args.source and self.args.target and self.args.model:
            self.logger.info("Start auto labelimg")
            self.input()
            self.process()
            self.output()
            self.logger.info("Done auto labelimg")
        else:
            self.parser.print_help()
            exit()

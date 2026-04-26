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
        self.report = []
        self.logger = logging.getLogger(__class__.__name__)

    def mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def input(self):
        if self.args.clean:
            if not self._confirm_clean(self.args.source, self.args.target):
                exit()
            if os.path.exists(self.args.target):
                shutil.rmtree(self.args.target)
        if self.args.val < 10 or self.args.val > 80:
            print(f"--val 超出范围: {self.args.val}，必须在 10~80% 之间")
            self.logger.error(f"--val out of range: {self.args.val}")
            exit()

        self.mkdirs(os.path.join(self.args.target))
        directory = [
            "train/labels",
            "train/images",
            "val/labels",
            "val/images",
            "test/labels",
            "test/images",
        ]

        classes = self.args.classes or os.path.join(self.args.source, "classes.txt")
        if not os.path.isfile(classes):
            print(f"classes.txt 文件不存在: {classes}")
            self.logger.error(f"classes.txt 文件不存在！")
            exit()
        else:
            with open(classes) as file:
                for line in file:
                    label = line.strip()
                    if not label:
                        continue
                    self.classes.append(label)
                    self.lables[label] = []
                self.logger.info(
                    f"classes len={len(self.classes)} labels={self.classes}"
                )
        if len(self.classes) == 0:
            print(f"classes.txt 没有有效标签: {classes}")
            self.logger.error(f"classes.txt empty labels: {classes}")
            exit()

        if not self.args.uuid:
            has_subdirs = any(
                os.path.isdir(os.path.join(self.args.source, name))
                for name in os.listdir(self.args.source)
            )
            if has_subdirs:
                self.logger.warning(
                    "source has subdirectories, output files may be overwritten by duplicate names, recommend --uuid"
                )
                print("source 存在子目录，输出文件可能因同名被覆盖，建议使用 --uuid 参数")
                try:
                    answer = input("是否继续？[y/N]: ").strip().lower()
                except EOFError:
                    answer = ""
                if answer not in ("y", "yes"):
                    print("已取消操作")
                    self.logger.info("cancel labelimg operation by subdirectory warning")
                    exit()

        files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)
        with tqdm(total=len(files), ncols=100) as progress:
            progress.set_description("file scanning")
            for source in files:
                progress.set_postfix_str(f"file={os.path.basename(source)[:36]:<36}")
                if source.endswith("classes.txt"):
                    self.missed.append((source, '忽略'))
                    progress.update(1)
                    continue
                if os.path.getsize(source) == 0:
                    self.missed.append((source, '.txt 空'))
                    self.logger.warning(f"标注文件为空: {source}")
                    progress.update(1)
                    continue
                for ext in Common.image_exts:
                    if os.path.exists(f"{os.path.splitext(source)[0]}{ext}"):
                        self.files[source] = f"{os.path.splitext(source)[0]}{ext}"
                        break
                else:
                    self.missed.append((source, "扩展名不支持"))
                    self.logger.warning(f"标注文件缺少配对图片: {source}")
                progress.update(1)

        with tqdm(total=len(directory), ncols=100) as progress:
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
                ncols=100,
                bar_format="{desc} {percentage:3.0f}%|{bar:58}| {n:>4.0f}/{total:>4.0f} {postfix}",
            ) as images,
            tqdm(
                total=len(self.files),
                ncols=100,
                bar_format="{desc} {percentage:3.0f}%|{bar:58}| {n:>4.0f}/{total:>4.0f} {postfix}",
            ) as train,
        ):
            for source in self.files.keys():

                train.set_description("train/labels")
                train.set_postfix_str(f"file={os.path.basename(source)[:36]:<36}")

                uuid4 = None
                if self.args.uuid:
                    uuid4 = uuid.uuid4()
                    label_target = os.path.join(
                        self.args.target, "train/labels", f"{uuid4}.txt"
                    )
                else:
                    label_target = os.path.join(
                        self.args.target, "train/labels", os.path.basename(source)
                    )
                name, extension = os.path.splitext(os.path.basename(label_target))

                valid_lines = []
                labels = []
                with open(source) as file:
                    for line_number, line in enumerate(file, start=1):
                        stripped = line.strip()
                        if not stripped:
                            self.missed.append((source, f"第 {line_number} 行为空"))
                            self.logger.error(f"empty label line file={source} line={line_number}")
                            continue

                        fields = stripped.split()
                        if len(fields) != 5:
                            self.missed.append((source, f"第 {line_number} 行格式错误"))
                            self.logger.error(f"invalid label format file={source} line={line_number} text={stripped}")
                            continue

                        try:
                            index = int(fields[0])
                        except ValueError:
                            self.missed.append((source, f"第 {line_number} 行类别非数字"))
                            self.logger.error(f"invalid label index file={source} line={line_number} index={fields[0]}")
                            continue

                        if index < 0:
                            self.missed.append((source, f"第 {line_number} 行类别为负数: {index}"))
                            self.logger.error(f"negative label index file={source} line={line_number} index={index}")
                            continue

                        if index >= len(self.classes):
                            self.missed.append((source, f"第 {line_number} 行类别越界: {index}"))
                            self.logger.error(f"label index out of range file={source} line={line_number} index={index}")
                            continue

                        try:
                            [float(value) for value in fields[1:]]
                        except ValueError:
                            self.missed.append((source, f"第 {line_number} 行坐标非数字"))
                            self.logger.error(f"invalid label coordinate file={source} line={line_number} text={stripped}")
                            continue

                        labels.append(self.classes[index])
                        valid_lines.append(stripped)

                if not valid_lines:
                    self.missed.append((source, "没有合法标签"))
                    self.logger.warning(f"标注文件没有合法标签: {source}")
                    train.update(1)
                    images.update(1)
                    continue

                with open(label_target, "w", encoding="utf-8") as file:
                    file.write("\n".join(valid_lines) + "\n")

                self.report.append((source, label_target))
                self.logger.debug(
                    f"train/labels source={source} target={label_target} name={name}"
                )
                train.update(1)
                # 图片复制
                images.set_description("train/images")
                image = self.files[source]
                images.set_postfix_str(f"file={os.path.basename(image)[:36]:<36}")

                if self.args.uuid:
                    image_target = os.path.join(
                        self.args.target,
                        "train/images",
                        f"{name}{os.path.splitext(image)[1]}",
                    )
                else:
                    image_target = os.path.join(
                        self.args.target, "train/images", os.path.basename(image)
                    )
                shutil.copy(image, image_target)
                self.logger.info(
                    f"train/images source={image} target={image_target} name={name}"
                )
                images.update(1)

                for label in labels:
                    self.lables[label].append(image_target)
                self.logger.info(f"file={image_target} labels={labels}")

        val_files = set()
        for label, files in self.lables.items():
            if len(files) == 0:
                continue
            files = list(dict.fromkeys(files))
            valnumber = int(len(files) * self.args.val / 100)
            if self.args.val > 0 and valnumber == 0:
                valnumber = 1
            if valnumber > len(files):
                valnumber = len(files)
            if valnumber == 0:
                continue

            val_files.update(random.sample(files, valnumber))
            # print(f"label={label} files={len(files)} val={len(vals)}")

        with tqdm(total=len(val_files), ncols=100) as progress:
            for file in sorted(val_files):
                progress.set_description("val")
                name, extension = os.path.splitext(os.path.basename(file))
                try:
                    source = os.path.join(
                        self.args.target, "train/labels", f"{name}.txt"
                    )
                    target = os.path.join(
                        self.args.target, "val/labels", f"{name}.txt"
                    )
                    if os.path.exists(source):
                        shutil.move(source, target)
                        self.logger.info(
                            f"val/labels move source={source} target={target}"
                        )
                    else:
                        self.logger.warning(f"val/labels missing train label name={name}")

                    source = file
                    target = os.path.join(
                        self.args.target, "val/images", os.path.basename(file)
                    )
                    if os.path.exists(source):
                        shutil.move(source, target)
                        self.logger.info(
                            f"val/images move source={source} target={target}"
                        )
                    else:
                        self.logger.warning(f"val/images missing train image name={name}")
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

        tables = [("丢失图像", "原因")]
        if self.missed:
            for file in self.missed:
                tables.append(file)

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
            tables.append([label, len(set(files))])
        table = Texttable(max_width=160)
        table.add_rows(tables)
        print(table.draw())

        if self.args.report:
            report_dir = os.path.dirname(self.args.report)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(self.args.report, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(('源文件', '目标文件'))
                writer.writerows(self.report)

    def main(self, args=None):
        if args is not None:
            self.args = args
        self.logger.info("Start")
        self.input()
        self.process()
        self.output()
        self.logger.info("Done")


class YoloLabelimgAutomatic(Common):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def __init__(self):
        self.basedir = BASE_DIR
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
                clean_paths = []
                for path in (self.args.target, self.args.output):
                    if path and path not in clean_paths:
                        clean_paths.append(path)
                if not self._confirm_clean(self.args.source, clean_paths):
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

    def main(self, args):
        self.args = args
        self.logger.info("Start auto labelimg")
        self.input()
        self.process()
        self.output()
        self.logger.info("Done auto labelimg")

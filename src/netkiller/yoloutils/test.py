import concurrent.futures
import csv
import glob
import logging
import os
import shutil

from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO

try:
    from . import Common
except ImportError:
    from __init__ import Common

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YoloTest:
    path = None
    model = None
    output = None
    tables = []

    def __init__(self, parser, args):
        self.logger = logging.getLogger(__class__.__name__)

        parser.add_argument('-s', '--source', type=str, default=None, help='图片来源地址')
        parser.add_argument('-t', '--target', type=str, default=None, help='图片目标地址')
        parser.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        parser.add_argument('-m', '--model', type=str, default=None, help='模型路径')
        parser.add_argument('-c', '--csv', type=str, default=None, help='保存结果', metavar="result.csv")
        parser.add_argument('-o', '--output', type=str, default=None, help='测试结果输出路径')
        parser.add_argument('-w', '--worker', type=int, default=1, help='线程数', metavar=1)

        self.parser = parser

        self.args = args
        self.yolo = None
        self.files = []
        self.total = 0

    def input(self):
        if self.args.clean and self.args.output:
            if os.path.exists(self.args.output):
                shutil.rmtree(self.args.output)
        if self.args.output:
            os.makedirs(self.args.output, exist_ok=True)

        files = glob.glob(f"{self.args.source}/**/*", recursive=True)
        self.files = [
            f
            for f in files
            if os.path.isfile(f) and f.lower().endswith(Common.image_exts)
        ]
        self.total = len(self.files)

    def process(self):
        self.tables = [["文件", "标签", "置信度"]]
        model = None
        try:
            model = YOLO(self.args.model)
        except FileNotFoundError as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e, f" {self.args.model}")
            exit()
        except Exception as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e)
            exit()

        max_workers = self.args.worker
        if max_workers < 1:
            max_workers = 1

        file_chunks = self._chunk_files(self.files, max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(max_workers):
                futures.append(
                    executor.submit(
                        self._process_worker,
                        model,
                        file_chunks[i],
                        i,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                thread_id, chunk_results = future.result()
                self.tables.extend(chunk_results)

    def _chunk_files(self, files, num_chunks):
        chunk_size = len(files) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            if i == num_chunks - 1:
                end = len(files)
            else:
                end = start + chunk_size
            chunks.append(files[start:end])
        return chunks

    def _process_worker(self, model, files, thread_id):
        results = []
        file_total = len(files)

        with tqdm(
                total=file_total,
                ncols=120,
                unit="file",
                mininterval=0.0,
                position=thread_id,
                desc=f"Thread-{thread_id}",
                leave=True,
        ) as progress:
            for file_index, file in enumerate(files, start=1):
                progress.set_postfix_str(os.path.basename(file))
                source, label, conf = self.detect(model, file)
                if conf is None:
                    conf = 0.0
                results.append([source, label, conf])
                progress.update(1)

        return thread_id, results

    def output(self):
        if self.args.csv:
            header = self.tables[0]
            rows = self.tables[1:]
            with open(self.args.csv, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)

        table = Texttable(max_width=160)
        table.add_rows(self.tables)
        print(table.draw())
        rows = self.tables[1:]
        if not rows:
            print("Total: 0, Not found: 0, Average: 0.00")
            return

        scores = [float(t[2]) for t in rows]
        average = sum(scores) / len(scores)
        miss = sum([t.count(None) for t in rows])
        print(
            f"Total: {self.total}, Not found: {miss}, Average: {'{:.2f}'.format(average)}"
        )

    def detect(self, model, source: str):
        if not source:
            return

        try:
            results = model.predict(source, verbose=False)  # , show_progress=True
            for result in results:
                boxes = result.boxes
                names = result.names
                if self.args.output:
                    filename = os.path.basename(result.path)
                    output = os.path.join(self.args.output, filename)
                    result.save(output)

                if names is not None:
                    for box in boxes:
                        label = names[int(box.cls)]
                        conf = "{:.2f}".format(float(box.conf))
                        return (source, label, conf)
        except Exception as e:
            self.logger.error(repr(e))
        return (source, None, None)

    def main(self):
        if self.args.source and self.args.model:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()


class YoloTestDiff:
    path = None
    model = None
    output = None
    tables = []

    def __init__(self, parser, args):
        self.logger = logging.getLogger(__class__.__name__)

        parser.add_argument('-s', '--source', type=str, default=None, help='图片来源地址')
        # parser.add_argument('-t', '--target', type=str, default=None, help='图片目标地址')
        parser.add_argument('--clean', action="store_true", default=False, help='清理之前的数据')
        # parser.add_argument('--diff', action="store_true", default=False, help='对比模型')
        parser.add_argument('-m', "--model", nargs="+", default=None, help="模型", metavar="best1.pt best2.pt best3.pt")
        parser.add_argument('-l', '--label', type=str, default=None, help='标签过滤只统计指定标签', metavar="")
        parser.add_argument('-o', '--output', type=str, default=None, help='对比结果输出路径')
        parser.add_argument('-c', '--csv', type=str, default=None, help='保存对比结果', metavar="result.csv")

        self.parser = parser
        self.args = args
        self.files = []
        self.total = 0
        self.models = {}
        self.scores = {}

    def input(self):

        if self.args.clean and self.args.output:
            if os.path.exists(self.args.output):
                shutil.rmtree(self.args.output)
        if self.args.output:
            os.makedirs(self.args.output, exist_ok=True)

        files = glob.glob(f"{self.args.source}/**/*", recursive=True)
        self.files = [
            f
            for f in files
            if os.path.isfile(f) and f.lower().endswith(Common.image_exts)
        ]
        self.total = len(self.files)

        header = ["文件", "标签"]

        try:
            for model in self.args.model:
                self.models[model] = YOLO(model)
                self.scores[model] = {}
                header.append(model)
            self.tables = [header]
        except FileNotFoundError as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e, f" {self.args.model}")
            exit()
        except Exception as e:
            self.logger.error(repr(e))
            print(type(e).__name__, ": ", e)
            exit()

    def process(self):
        files = [os.path.abspath(file) for file in self.files]
        labels = {file: set() for file in files}
        model_items = list(self.models.items())
        workers = max(1, len(model_items))
        source_prefix = ""
        if self.args.source:
            source_prefix = (
                    os.path.normpath(os.path.abspath(self.args.source)) + os.sep
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for position, (name, model) in enumerate(model_items):
                futures.append(
                    executor.submit(
                        self._process_model,
                        name,
                        model,
                        files,
                        position,

                    )
                )

            for future in concurrent.futures.as_completed(futures):
                name, model_scores, model_labels = future.result()
                self.scores[name] = model_scores
                for file, found in model_labels.items():
                    if found:
                        labels[file].update(found)

        for file in files:
            found = sorted(labels.get(file, set()))
            if found:
                label_text = ",".join(found)
            else:
                label_text = ""
            filename = file.replace(source_prefix, "") if source_prefix else file
            column = [filename, label_text]
            for name in self.models:
                column.append(self.scores[name].get(file, 0.0))
            self.tables.append(column)

    def _process_model(self, name, model, files, position):
        model_scores = {}
        model_labels = {file: set() for file in files}
        file_total = len(files)
        model_name = os.path.basename(name)

        with tqdm(
                total=file_total,
                ncols=120,
                unit="file",
                mininterval=0.0,
                position=position,
                desc=f"{model_name}",
                leave=True,
        ) as progress:
            for file_index, file in enumerate(files, start=1):

                filename = file.replace(os.path.normpath(os.path.abspath(self.args.source)) + os.sep, '')

                conf = 0.0
                found_labels = set()
                progress.set_postfix_str(filename)

                try:
                    results = model.predict(file, verbose=False)
                    for result in results:
                        if self.args.output:
                            output = os.path.join(self.args.output, model_name, filename)

                            os.makedirs(os.path.dirname(output), exist_ok=True)
                            result.save(output)
                            self.logger.info(f"output {output}")

                        names = result.names
                        boxes = result.boxes
                        if names is not None:
                            for box in boxes:
                                label = names[int(box.cls)]
                                if self.args.label and label != self.args.label:
                                    continue
                                found_labels.add(label)
                                conf = max(conf, float(box.conf))
                except Exception as e:
                    self.logger.error(repr(e))

                if found_labels:
                    model_labels[file].update(found_labels)
                model_scores[file] = "{:.2f}".format(conf)
                progress.update(1)

        return name, model_scores, model_labels

    def output(self):
        if self.args.csv:
            header = self.tables[0]
            rows = self.tables[1:]
            with open(self.args.csv, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)

        table = Texttable(max_width=160)
        table.add_rows(self.tables)
        print(table.draw())
        rows = self.tables[1:]
        if not rows:
            summary = ["Total: 0"]
            for model in self.tables[0][2:]:
                summary.append(f"Average({os.path.basename(model)}): 0.00")
            print(", ".join(summary))
            return

        averages = []
        for index, model in enumerate(self.tables[0][2:], start=2):
            model_scores = []
            for row in rows:
                try:
                    model_scores.append(float(row[index]))
                except (TypeError, ValueError):
                    model_scores.append(0.0)
            average = sum(model_scores) / len(model_scores)
            averages.append(f"{os.path.basename(model)}={average:.2f}")
        print(f"Total: {self.total} Average: {', '.join(averages)}")

    def main(self):
        if self.args.source and self.args.model and len(self.args.model) > 0:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()

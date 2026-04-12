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


class YoloTest:
    path = None
    model = None
    output = None
    tables = []

    def __init__(self, parser, args):
        self.logger = logging.getLogger(__class__.__name__)
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

        with tqdm(total=self.total, ncols=100) as progress:
            for file in self.files:
                progress.set_description("%s" % file)
                source, label, conf = self.detect(model, file)
                if conf is None:
                    conf = 0.0
                self.tables.append([os.path.basename(source), label, conf])
                progress.update(1)

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
            results = model.predict(source, verbose=False)
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
        self.parser = parser
        self.args = args
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
        header = ["文件", "标签"]
        models = {}
        scores = {}
        try:
            for model in self.args.models:
                models[model] = YOLO(model)
                scores[model] = {}
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

        files = [os.path.abspath(file) for file in self.files]
        labels = {file: set() for file in files}
        model_items = list(models.items())
        model_total = len(model_items)
        file_total = len(files)
        total_tasks = model_total * file_total
        with tqdm(total=total_tasks, ncols=150, unit="task", mininterval=0.0) as progress:
            for model_index, (name, model) in enumerate(model_items, start=1):
                model_name = os.path.basename(name)
                progress.set_description(
                    f"model={model_index}/{model_total}({model_name})"
                )
                for file_index, file in enumerate(files, start=1):
                    filename = os.path.basename(file)
                    conf = 0.0
                    found_labels = set()
                    progress.set_postfix_str(
                        f"file={file_index}/{file_total} {filename}"
                    )

                    try:
                        results = model.predict(file, verbose=False)
                        for result in results:
                            if self.args.output:
                                output = os.path.join(self.args.output, filename)
                                result.save(output)

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
                        labels[file].update(found_labels)
                    scores[name][file] = "{:.2f}".format(conf)
                    progress.update(1)

        for file in files:
            found = sorted(labels.get(file, set()))
            if found:
                label_text = ",".join(found)
            else:
                label_text = ""
            column = [os.path.basename(file), label_text]
            for name in models:
                column.append(scores[name].get(file, 0.0))
            self.tables.append(column)

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

        summary = [f"Total: {self.total}"]
        for index, model in enumerate(self.tables[0][2:], start=2):
            model_scores = []
            for row in rows:
                try:
                    model_scores.append(float(row[index]))
                except (TypeError, ValueError):
                    model_scores.append(0.0)
            average = sum(model_scores) / len(model_scores)
            summary.append(f"Average({os.path.basename(model)}): {average:.2f}")
        print(", ".join(summary))

    def main(self):
        if self.args.source and self.args.models and len(self.args.models) > 0:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()

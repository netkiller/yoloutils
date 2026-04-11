import csv
import glob
import logging
import os
import shutil

from texttable import Texttable
from tqdm import tqdm
from ultralytics import YOLO


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
            f for f in files if os.path.isfile(f) and not f.endswith((".txt", ".DS_Store"))
        ]
        self.total = len(self.files)

    def test(self):
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

    def diff(self):
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
        with tqdm(total=len(models), ncols=150) as progress:
            for name, model in models.items():
                progress.set_description(f"model={name}")
                try:
                    for result in model.predict(files, verbose=False, stream=True):
                        source = os.path.abspath(result.path)
                        conf = 0.0

                        if self.args.output:
                            filename = os.path.basename(result.path)
                            output = os.path.join(self.args.output, filename)
                            result.save(output)

                        names = result.names
                        boxes = result.boxes
                        if names is not None:
                            for box in boxes:
                                conf = "{:.2f}".format(float(box.conf))
                                break
                        scores[name][source] = conf
                except Exception as e:
                    self.logger.error(repr(e))
                progress.update(1)

        for file in files:
            column = [os.path.basename(file), self.args.label]
            for name in models:
                column.append(scores[name].get(file, 0.0))
            self.tables.append(column)

    def process(self):
        if self.args.diff:
            self.diff()
        else:
            self.test()

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

        if self.args.diff:
            scores = [float(conf) for row in rows for conf in row[2:]]
        else:
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
        if self.args.diff and self.args.source and self.args.models and len(self.args.models) > 0:
            self.input()
            self.process()
            self.output()
        elif self.args.source and self.args.model:
            self.input()
            self.process()
            self.output()
        else:
            self.parser.print_help()
            exit()

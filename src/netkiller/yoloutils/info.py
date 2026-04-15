import os

from ultralytics import YOLO


class YoloInfo:
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        pass

    def info(self):
        model = YOLO(self.model)
        params = model.ckpt["train_args"]
        for key, value in params.items():
            print(f"{key}: {value}")

    def f1(self):
        model = YOLO(self.model)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_metrics = ckpt.get("train_metrics", {})
        precision = train_metrics.get("metrics/precision(B)")
        recall = train_metrics.get("metrics/recall(B)")

        # 优先使用 checkpoint 里已记录的 precision/recall 计算 F1，避免依赖外部数据集路径。
        if precision is not None and recall is not None:
            denominator = precision + recall
            mean_f1 = (2 * precision * recall / denominator) if denominator else 0.0
            print(
                f"F1: {mean_f1:.5f} (precision={precision:.5f}, recall={recall:.5f}, source=checkpoint)"
            )
            return

        # 如果 checkpoint 不含训练指标，尝试在可用 data 上临时验证。
        train_args = ckpt.get("train_args", {})
        data = train_args.get("data")
        if data and os.path.exists(data):
            metrics = model.val(
                data=data,
                project="/tmp",
                name="yoloutils-info-f1",
                exist_ok=True,
                plots=False,
                save=False,
                verbose=False,
            )
            f1 = metrics.box.f1
            mean_f1 = metrics.box.mf1
            print(f"各类别: {f1}, 所有类别平均: {mean_f1}")
            return

        print("无法计算 F1：checkpoint 缺少 precision/recall，且 data 路径不可用。")

    def recall(self):
        model = YOLO(self.model)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_metrics = ckpt.get("train_metrics", {})
        recall = train_metrics.get("metrics/recall(B)")

        # 优先使用 checkpoint 中的召回率，避免依赖外部数据集路径。
        if recall is not None:
            print(f"Recall: {recall:.5f} (source=checkpoint)")
            return

        # 如果 checkpoint 不含训练指标，尝试在可用 data 上临时验证。
        train_args = ckpt.get("train_args", {})
        data = train_args.get("data")
        if data and os.path.exists(data):
            metrics = model.val(
                data=data,
                project="/tmp",
                name="yoloutils-info-recall",
                exist_ok=True,
                plots=False,
                save=False,
                verbose=False,
            )
            print(f"Recall: {metrics.box.mr:.5f} (source=val)")
            return

        print("无法计算 Recall：checkpoint 缺少 recall，且 data 路径不可用。")

    def precision(self):
        model = YOLO(self.model)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_metrics = ckpt.get("train_metrics", {})
        precision = train_metrics.get("metrics/precision(B)")

        # 优先使用 checkpoint 中的精确率，避免依赖外部数据集路径。
        if precision is not None:
            print(f"Precision: {precision:.5f} (source=checkpoint)")
            return

        # 如果 checkpoint 不含训练指标，尝试在可用 data 上临时验证。
        train_args = ckpt.get("train_args", {})
        data = train_args.get("data")
        if data and os.path.exists(data):
            metrics = model.val(
                data=data,
                project="/tmp",
                name="yoloutils-info-precision",
                exist_ok=True,
                plots=False,
                save=False,
                verbose=False,
            )
            print(f"Precision: {metrics.box.mp:.5f} (source=val)")
            return

        print("无法计算 Precision：checkpoint 缺少 precision，且 data 路径不可用。")

    def accuracy(self):
        model = YOLO(self.model)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_metrics = ckpt.get("train_metrics", {})

        # 检测模型通常无“accuracy”，这里以 mAP50 作为准确率指标输出。
        accuracy = train_metrics.get("metrics/mAP50(B)")
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.5f} (metric=mAP50, source=checkpoint)")
            return

        # 兼容分类模型常见 top1 指标。
        top1 = train_metrics.get("metrics/accuracy_top1")
        if top1 is not None:
            print(f"Accuracy: {top1:.5f} (metric=top1, source=checkpoint)")
            return

        # 如果 checkpoint 不含训练指标，尝试在可用 data 上临时验证。
        train_args = ckpt.get("train_args", {})
        data = train_args.get("data")
        if data and os.path.exists(data):
            metrics = model.val(
                data=data,
                project="/tmp",
                name="yoloutils-info-accuracy",
                exist_ok=True,
                plots=False,
                save=False,
                verbose=False,
            )
            if hasattr(metrics, "box") and hasattr(metrics.box, "map50"):
                print(f"Accuracy: {metrics.box.map50:.5f} (metric=mAP50, source=val)")
                return
            if hasattr(metrics, "top1"):
                print(f"Accuracy: {metrics.top1:.5f} (metric=top1, source=val)")
                return

        print("无法计算 Accuracy：checkpoint 缺少准确率指标，且 data 路径不可用。")

    def mAP(self):
        model = YOLO(self.model)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_metrics = ckpt.get("train_metrics", {})
        map50 = train_metrics.get("metrics/mAP50(B)")
        map5095 = train_metrics.get("metrics/mAP50-95(B)")

        # 优先使用 checkpoint 中的 mAP 指标，避免依赖外部数据集路径。
        if map50 is not None and map5095 is not None:
            print(
                f"mAP50: {map50:.5f}, mAP50-95: {map5095:.5f} (source=checkpoint)"
            )
            return

        # 如果 checkpoint 不含训练指标，尝试在可用 data 上临时验证。
        train_args = ckpt.get("train_args", {})
        data = train_args.get("data")
        if data and os.path.exists(data):
            metrics = model.val(
                data=data,
                project="/tmp",
                name="yoloutils-info-map",
                exist_ok=True,
                plots=False,
                save=False,
                verbose=False,
            )
            print(
                f"mAP50: {metrics.box.map50:.5f}, mAP50-95: {metrics.box.map:.5f} (source=val)"
            )
            return

        print("无法计算 mAP：checkpoint 缺少 mAP 指标，且 data 路径不可用。")

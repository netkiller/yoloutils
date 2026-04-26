import hashlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Common:
    # background = (22, 255, 39) # 绿幕RGB模式（R22 - G255 - B39），CMYK模式（C62 - M0 - Y100 - K0）
    background = (0, 0, 0)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif", ".avif")

    def __init__(self):
        self.logger = None
        self.basedir = BASE_DIR
        sys.path.append(self.basedir)

    def mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _confirm_clean(self, source, target):
        source_path = os.path.abspath(source)
        targets = target if isinstance(target, (list, tuple, set)) else [target]
        targets = [path for path in targets if path]

        for path in targets:
            target_path = os.path.abspath(path)
            if source_path == target_path:
                print("--target 不能与 --source 相同")
                if self.logger:
                    self.logger.error("--target same as --source")
                exit()
            if os.path.commonpath([source_path, target_path]) == target_path:
                print("--target 不能是 --source 的父目录")
                if self.logger:
                    self.logger.error("--target parent of --source")
                exit()

        existing_targets = [path for path in targets if os.path.exists(path)]
        if not existing_targets:
            return True

        print("检测到 --clean，将删除以下目录：")
        for path in existing_targets:
            print(f"- {path}")
        try:
            answer = input("是否继续？[y/N]: ").strip().lower()
        except EOFError:
            answer = ""

        if answer in ("y", "yes"):
            return True

        print("已取消清理操作")
        if self.logger:
            self.logger.info("cancel clean operation by user")
        return False

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

    def walkdir(self, path):
        for dirpath, dirnames, filenames in os.walk(path):
            print(f"dirpath={dirpath}, dirnames={dirnames}, filenames={filenames}")

    def md5sum(self, filename):
        md5 = hashlib.md5()
        with open(filename, "rb") as f:
            md5.update(f.read())
            return md5.hexdigest()


def main():
    from .yoloutils import main as _main

    return _main()


__all__ = ["BASE_DIR", "Common", "main"]

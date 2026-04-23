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

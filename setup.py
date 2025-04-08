import os, sys
from setuptools import setup, find_packages

with open("README.md", "r",encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="yoloutils",
    version="1.0.0",
    author="Neo Chen",
    author_email="netkiller@msn.com",
    description="Yolo labels Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.netkiller.cn/python/index.html",
    license="BSD",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "ultralytics",
        "onnx",
        "numpy",
        "matplotlib",
        "opencv-python",
        "pillow",
        "texttable",
        "dominate"
    ],
    # package_dir={ '': 'library' },
    packages=find_packages(),
    scripts=[
        "bin/yoloutils",
    ],
    data_files=[

    ],
)

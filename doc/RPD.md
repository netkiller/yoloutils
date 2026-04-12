# YoloLabelimg 优化需求

def process(self): 函数，中的

files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)

移动到 def input(self): 中

然后交验检查 files，除了 classes.txt 文件，其他.txt文件，必须有一个配对图片（图片扩展名包括 Common image_exts）

不要定义多余的新变量变量，处理范围是 input(self): 函数。其他代码不要动。

最后统计 有多少 .txt 文件。有多少 只有.txt 文件，丢失图像的数量。 用字符串打印在报表最下方。




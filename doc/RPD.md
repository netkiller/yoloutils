# YoloLabelimg 优化需求

def process(self): 函数，中的

files = glob.glob(f"{self.args.source}/**/*.txt", recursive=True)

移动到 def input(self): 中

然后交验检查 files，除了 classes.txt 文件，其他.txt文件，必须有一个配对图片（图片扩展名包括 Common image_exts）

不要定义多余的新变量变量，处理范围是 input(self): 函数。其他代码不要动。



print(f"标注文件缺少配对图片: {source}") 改为，统计 .txt 文件丢失图像的数量。 

最后用字符串打印在报表最下方。
+----------+------+
|   标签   | 数量 |
+==========+======+
| chongcao | 977  |
+----------+------+

+----------+
|   丢失图像   |
+==========+
| xxx.txt |
+----------+
Total: 1211, Lost: 222



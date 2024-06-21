---
typora-root-url: imgs
---

# 基于Python OpenCV进行印章分割

利用了计算机视觉和机器学习技术，通过颜色聚类和图像处理技术来自动化印章的提取。



![](result_01.png)

![](result_02.png)

![](result_03.png)



**使用方法：**

```
python main.py --help
usage: main.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

Batch extract seals with dominant color from images.

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the folder containing input images.
  --output_dir OUTPUT_DIR
                        Path to the folder for saving output images.
```

**示例：**

`python main.py --input_dir path_to_your_input_dir --output_dir path_to_your_output_dir`

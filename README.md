---
typora-root-url: imgs
---

# Seal Segmentation Based on Python OpenCV

中文说明见：READM_ZH-CN.md

Utilizing computer vision and machine learning technologies, this process automates the extraction of seals through color clustering and image processing techniques.



![](result_01.png)

![](result_02.png)

![](result_03.png)



**Usage:**

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

**Example:**

`python main.py --input_dir path_to_your_input_dir --output_dir path_to_your_output_dir`

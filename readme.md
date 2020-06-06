## 文件说明：

src目录下：

```
.
├── .DS_Store
├── MYHRNet.py 修改过网络结构的hrnet
├── __init__.py
├── __pycache__
│   ├── dataset.cpython-36.pyc
│   └── heatmap.cpython-36.pyc
├── configuration
│   ├── .DS_Store
│   ├── __init__.py
│   ├── base_config.py 训练的基本配置参数
│   ├── coco_conf
│   │   ├── __init__.py
│   │   ├── coco_annotation.py 将coco keypoint的json数据读取 转化为txt，方便TextLineDataset直读取
│   │   └── make_dataset.py 制作数据集，但每个数据仅仅是txt文本里的一行
│   └── write_coco_to_txt.py 
├── dataset.py 没有用到
├── evaluate.py 没有用到（因为还没改test
├── heatmap.py 没有用到
├── high_resolution_module.py 包含修改后的网络模块 BasicBlock bottleblock 和后面的并行模块，以及多尺度融合模块 
├── hrnet.py 原仓库的网络模型
├── make_ground_truth.py 将数据集（make_dataset）中的每行信息进行读取，生成输入图像，以及gt的heatmap
├── model_module.py  原仓库的的网络模块
├── myTrain.py 训练的main函数
├── temp.py 没用到
├── test.py 暂时没用到
├── train.py 没用到
└── utils
    ├── __init__.py
    └── transforms.py 用来对原始图像进行处理，包括crop，缩放等
```


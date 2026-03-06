## 🌾 水稻病害智能检测系统 (Rice Disease Detection)

​	基于 **YOLOv8** 深度学习框架打造的水稻病害自动识别系统。本项目旨在帮助农户和研究人员快速、准确地识别三种常见的水稻叶片病害：**褐斑病 (Brown Spot)**、**稻瘟病 (Rice Blast)** 和 **白叶枯病 (Bacterial Blight)**。

​	本项目包含完整的训练代码、评估脚本以及一个交互式的 Web 演示界面。



## ✨ 主要功能

- **🖼️ 智能识别**：上传图片，自动框选病害区域并分类。
- **📊 置信度分析**：实时显示检测结果的置信度，辅助专业判断。
- **💻 Web 界面**：简洁友好的操作界面，支持检测结果图片一键下载。
- **🎓 迁移学习**：基于 COCO 预训练权重，即使在少量数据下也能快速收敛。



## 🚀 从零开始搭建指南（终端运行版）

只需 5 分钟，即可在本地运行此项目。请严格按顺序执行以下步骤。

### 1️⃣ 环境准备与代码获取

首先，确保你的电脑已安装 **Git** 和 **Python 3.8+**。然后打开终端（CMD / PowerShell / Terminal），执行以下命令克隆项目：

```bash
# 克隆项目到本地
git clone https://github.com/yuanzhenX/rice-disease-detection.git

# 进入项目目录
cd rice-disease-detection
```

### ️2️⃣安装依赖环境

为了隔离环境避免冲突，我们需要创建一个虚拟环境并安装所需的 Python 库。请在项目目录下依次执行：

```bash
# 创建虚拟环境 (名为 venv)
python -m venv venv

# 激活虚拟环境(Windows环境)
venv\Scripts\activate

# 安装项目所需的所有依赖库
pip install -r requirements.txt
```

###  3️⃣准备模型与数据

准备好数据集，确保目录结构为 datasets/images 和 datasets/labels。
修改 data.yaml 文件中的路径指向你的数据集。
运行训练脚本：

```bash
python train.py
```

*训练完成后，最佳模型会自动保存，无需手动移动文件。*

### 4️⃣启动 Web 系统

一切就绪后，确保虚拟环境已激活，运行以下命令启动界面：

```bash
streamlit run app.py
```


系统会自动在浏览器中打开 http://localhost:8501。
上传一张水稻叶片图片，即可看到检测结果！

## 📂 项目结构说明

```text
rice-disease-detection/
├── app.py                # Streamlit Web 应用主程序
├── train_model.py        # 模型训练脚本
├── rice_data.yaml        # 数据集配置文件
├── requirements.txt      # Python 依赖列表
├── README.md             # 项目说明文档
├── .gitignore            # Git 忽略文件配置
├── runs/                 # [运行训练脚本后生成] 训练输出目录
│   └── detect/.../weights/best.pt
└── datasets/             # 数据集存放目录
    ├── images/
    └── labels/
```



## 🚀 从零开始搭建指南（PyCharm + Anaconda 版）

本教程专为 **PyCharm** 用户设计，配合 **Anaconda** 管理环境，全程图形化操作，简单直观。

### 1️⃣ 配置 PyCharm 与 Anaconda 环境

**前提**：确保已安装 [Anaconda](https://www.anaconda.com/) 和 [PyCharm](https://www.jetbrains.com/pycharm/)。

####  第一步：创建 Conda 虚拟环境

​	请自主学习B站视频【YOLO环境配置】https://www.bilibili.com/video/BV182bZzMEYD?vd_source=d61a5eaf9a6f3e9647152d9e23fc6ddb

####  第二步：在 PyCharm 中打开项目 

1. 启动 PyCharm，点击 **Open**。 
2. 选择你克隆下来的 `rice-disease-detection` 文件夹。

#### 第三步：安装依赖库

. 等待环境创建完成后，确保当前解释器显示为新创建的 `conda-env`。
. 点击解释器列表下方的 **+ Add Package** 按钮。
. 在搜索框输入以下库名并依次点击 **Install Package**：

   - `ultralytics`
   - `streamlit`
   - `opencv-python`
   - `pillow`
   - `matplotlib`
   - `seaborn`

*(或者，如果项目根目录有 `requirements.txt`，可直接点击底部的 **Install requirements.txt**)*

> **✅ 环境搭建成功标志**：PyCharm 底部状态栏显示的 Python 版本是你刚才创建的 Conda 环境，且没有红色波浪线报错。



###  2️⃣ 核心功能操作指南

#### ① 训练模型 (Train) 

*目的：让模型学习水稻病害特征。* 

**操作步骤**：

1. 确保数据集已放入 `datasets/` 文件夹，且 `data.yaml` 中的路径配置正确

2. 运行 `train.py`

3. 经过训练后，目录会出现runs/detect/rice_exp_01文件夹，内部包含

   -  `weights/best.pt` (最佳模型权重)  

   - `results.png` (损失函数和精度变化曲线图) 
   -  `confusion_matrix.png` (混淆矩阵)

#### ② 使用模型预测 (Predict) 

*目的：单张图片测试模型效果。*

**操作步骤**：

 	1. 运行predict_model.py
 	2. 在runs中出现的predict文件夹中查看预测结果

####  ③ 评估模型 (Validate) 

*目的：量化模型在验证集上的表现（mAP, Precision, Recall）。*

**操作步骤**：

 	1. 运行eval_model.py
 	2. 在运行框查看结果
 	3. 在runs中出现的val文件夹中查看评估图像

#### ④ 打开 Web 网页 (Launch UI) 

*目的：启动交互式演示系统，供非技术人员使用。*

**操作步骤**：

 1. 在pycharm终端运行以下代码

    ```bash
    streamlit run app.py
    ```

    

 2. 若没有自动跳转到网页，请手动浏览器访问*http://localhost:8501*

 3. 按照网页提示进行操作



## ⚙️ 常见问题 (FAQ)

**Q: 运行时提示找不到 best.pt 怎么办？**
A: 请检查 app.py 中加载模型的路径是否正确，或者先执行“步骤 3”中的训练命令生成模型。

**Q: 如何修改检测的类别？**
A: 编辑 data.yaml 文件中的 names 列表，并重新运行 train.py 进行训练。

**Q: 想要更高的检测精度？**
A: 在 train.py 中将模型架构从 yolov8n (Nano) 改为 yolov8s (Small) 或 yolov8m (Medium)，但这会增加训练时间。

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request 来改进这个项目！
如果你在使用过程中遇到问题，请查看 Issues 板块。

## 📄 许可证

本项目采用 MIT 许可证开源。
Made with ❤️ by yuanzhenX

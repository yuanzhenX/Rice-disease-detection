"""
该代码作用如下：
基于数据集的训练集和验证集对模型yolov8n进行训练、微调
从而达到识别病变的效果
"""

from ultralytics import YOLO
import os


def train():
    # 1. 加载模型
    # 如果根目录有 yolov8n.pt 会用本地的，没有会自动下载
    model = YOLO('yolov8n.pt')

    # 2. 配置文件路径
    data_config = 'rice_data.yaml'

    # 双重检查文件是否存在，防止路径错误
    if not os.path.exists(data_config):
        print(f"严重错误：找不到配置文件 {os.path.abspath(data_config)}")
        print("请确认 rice_data.yaml 是否在项目根目录下！")
        return

    print(f"配置文件已找到：{data_config}")
    print("开始训练...")

    # 3. 开始训练
    results = model.train(
        data=data_config,
        epochs=100,
        imgsz=320,
        batch=16,  # 如果显存爆 (CUDA out of memory)，请改为 8 或 4
        device=0,  # 0 代表 GPU，如果没有 GPU 改为 'cpu'
        workers=0,  # 【关键修改】Windows 下建议设为 0，避免多进程报错
        project='runs/detect',
        name='rice_exp_01'
    )

    print("🎉训练完成！结果保存在 runs/detect/rice_exp_01")


if __name__ == '__main__':
    train()

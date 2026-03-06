"""
该文件用于使用训练好的模型预测测试集的内容并进行保存
"""
from ultralytics import YOLO
import os

# 1. 加载训练好的最佳模型
model_path = 'runs/detect/runs/detect/rice_exp_01/weights/best.pt'

if not os.path.exists(model_path):
    print(f"找不到模型文件：{model_path}")
    print("请检查 runs/detect/... 文件夹下的文件名是否正确")
else:
    model = YOLO(model_path)
    print(f"模型加载成功：{model_path}")

    # 2. 准备测试图片
    source_folder = 'datasets/RiceDiseaseDataset_yolo_test/images/test'

    if not os.path.exists(source_folder):
        # 如果没有测试文件夹，临时用验证集代替
        source_folder = 'datasets/RiceDiseaseDataset_yolo_test/images/valid'
        print(f" 未找到 {source_folder}，尝试使用验证集图片进行演示...")

    print(f" 开始对 {source_folder} 中的图片进行预测...")

    # 3. 执行预测
    results = model.predict(
        source=source_folder,
        save=True,  # 保存带有检测框的图片
        save_txt=False,  # 是否保存标签txt文件
        conf=0.2,  # 置信度阈值，低于0.25的框不显示（可调高或调低）
        iou=0.45,  # NMS阈值
        show=False  # 是否在屏幕上弹窗显示（服务器运行建议设为False）
    )

    # 4. 查看结果
    output_dir = 'runs/detect/predict'   # 预测结果图片保存在：runs/detect/predict/
    print(f"预测完成！带框的结果图片已保存在：{os.path.abspath(output_dir)}")
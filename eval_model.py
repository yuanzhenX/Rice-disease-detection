"""
该代码用于评估模型
"""
from ultralytics import YOLO
import os


def main():
    print("🚀 开始水稻病害模型评估...")

    # 1. 模型路径
    model_path = 'runs/detect/runs/detect/rice_exp_01/weights/best.pt'

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件！\n路径：{model_path}")
        print("请确认你是否已经完成了训练，并且模型保存在该位置。")
        return

    # 2. 加载模型
    print(f"✅ 正在加载模型：{model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 配置文件路径
    data_yaml = 'rice_data.yaml'

    if not os.path.exists(data_yaml):
        print(f"❌ 错误：找不到配置文件 {data_yaml}")
        print("请确保 rice_data.yaml 在当前目录下，且内容已正确修改。")
        return

    print(f"✅ 使用配置文件：{os.path.abspath(data_yaml)}")

    # 4. 执行评估 (标准调用)
    # split='val' 对应 yaml 中的 val 字段
    # plots=True 会生成混淆矩阵等图表
    print("📊 正在运行验证集评估 (这可能需要几分钟)...")
    try:
        metrics = model.val(data=data_yaml, split='val', plots=True, verbose=True)

        # 5. 输出结果
        print("\n" + "=" * 60)
        print("🎉 评估完成！结果如下：")
        print("=" * 60)
        print(f"🎯 mAP50 (IoU=0.50):    {metrics.box.map50:.4f}  ({metrics.box.map50 * 100:.2f}%)")
        print(f"🎯 mAP50-95 (IoU=0.5:0.95): {metrics.box.map:.4f}  ({metrics.box.map * 100:.2f}%)")
        print(f"📈 精确率 (Precision):   {metrics.box.mp:.4f}")
        print(f"📉 召回率 (Recall):      {metrics.box.mr:.4f}")
        print("=" * 60)
        print(f"💡 详细图表已保存至：runs/detect/val/")

    except Exception as e:
        print(f"\n❌ 评估过程中发生严重错误：{e}")
        print("\n💡 建议排查步骤：")
        print("1. 检查 rice_data.yaml 中的路径是否真实存在。")
        print("2. 检查类别数量是否与模型训练时一致 (当前设为 3 类)。")
        print("3. 尝试删除 runs/detect/val 文件夹后重试。")


if __name__ == '__main__':
    main()
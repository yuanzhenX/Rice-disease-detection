"""
该代码用于创建网页
"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import io

# 页面配置
st.set_page_config(page_title="水稻病害检测系统", page_icon="🌾", layout="wide")

st.title("🌾 水稻病害智能检测系统")
st.markdown("""
基于 **YOLOv8** 深度学习模型，支持识别三种常见水稻病害：
- **Brown Spot** (褐斑病)
- **Rice Blast** (稻瘟病)
- **Bacterial Blight** (白叶枯病)
""")

# 侧边栏：模型加载
st.sidebar.header("⚙️ 设置")
model_path = "runs/detect/runs/detect/rice_exp_01/weights/best.pt"


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    else:
        return None


model = load_model(model_path)

if model is None:
    st.error(f"❌ 模型文件未找到：{model_path}\n请确认模型已训练并保存。")
    st.stop()

st.sidebar.success("✅ 模型加载成功")

# 主界面：上传与检测
uploaded_file = st.file_uploader("📤 上传水稻叶片图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原图
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图片")
        image = Image.open(uploaded_file)
        st.image(image, width='stretch')

    # 执行检测
    with st.spinner("🔍 正在分析病害..."):
        # 将上传的文件保存到临时文件，因为 YOLO 需要文件路径
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # 预测
            results = model.predict(source=tmp_path, conf=0.25, save=False, verbose=False)
            result = results[0]

            # 获取带框的图片 (Plotting)
            plot_img = result.plot()  # 返回的是 numpy array (BGR)
            plot_img_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)  # 转回 RGB 供 Streamlit 显示

            # 解析结果
            boxes = result.boxes
            detections = []
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    detections.append(f"**{name}**: {conf:.2%}")
            else:
                detections.append("未检测到明显病害。")

            # 显示结果图
            with col2:
                st.subheader("检测结果")
                # ✅ 修改点 2: 使用 use_container_width 替代 use_column_width
                st.image(plot_img_rgb, width='stretch')

            # 显示详细列表
            st.subheader("📋 详细分析")
            for det in detections:
                st.markdown(f"- {det}")

            # 下载按钮
            buf = io.BytesIO()
            pil_res = Image.fromarray(plot_img_rgb)
            pil_res.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 下载检测结果图",
                data=byte_im,
                file_name="detection_result.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"检测出错：{e}")
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    st.info("👆 请在上方上传一张图片开始检测。")

# 页脚
st.markdown("---")
st.caption("Powered by YOLOv8 & Streamlit | Rice Disease Detection Project")
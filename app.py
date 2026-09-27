"""
Streamlit Blob Detection API
Detects anatomical landmarks in body images
"""

import os

os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ["MEDIAPIPE_CACHE_DIR"] = "/tmp/mediapipe_cache"
os.environ["TMPDIR"] = "/tmp"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pillow_heif
from io import BytesIO
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register HEIF opener
pillow_heif.register_heif_opener()

# Import blob detection functions
from ex3_blob_detection import (
    find_blobs,
    deduplicate_points,
    separate_appropriate_points,
    compute_percentage,
    LABELS
)

# Page config
st.set_page_config(
    page_title="Blob Detection",
    page_icon="🎯",
    layout="wide"
)


def process_image(image_input):
    """Process image - detect blob landmarks"""
    try:
        # # Convert to numpy
        # image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        # h, w, _ = image.shape
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # # Pose detection
        # mp_pose = mp.solutions.pose
        # with mp_pose.Pose(
        #         static_image_mode=True,
        #         min_detection_confidence=0.5,
        #         model_complexity=1,
        #         enable_segmentation=True
        # ) as pose:
        #     results = pose.process(image_rgb)
        #
        # if not results.pose_landmarks:
        #     return None, {"error": "❌ No body detected. Try a clearer photo from behind."}
        #
        # # Segmentation
        # if results.segmentation_mask is not None:
        #     best_mask = (results.segmentation_mask > 0.8).astype(np.uint8) * 255
        # else:
        #     best_mask = np.ones((h, w), dtype=np.uint8) * 255
        #
        # foreground = np.where(best_mask[..., None], image, 0)
        #
        # # Get head points
        # head_points = []
        # if results.pose_landmarks:
        #     landmarks = results.pose_landmarks.landmark
        #     for idx, lm in enumerate(landmarks):
        #         px, py = int(lm.x * w), int(lm.y * h)
        #         head_points.append([px, py])
        #         if idx >= 10:
        #             break
        #
        # average_head_point = np.array(head_points).mean(axis=0) if head_points else np.array([w // 2, h // 2])
        #
        # # Find blobs
        # res = find_blobs(foreground)
        # deduplicated_points = deduplicate_points(np.array(res), min_cluster_size=20)
        # separated, pairs, midpoints = separate_appropriate_points(deduplicated_points)
        #
        # output = image.copy()
        #
        # if len(separated) != 12:
        #     return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), {
        #         "error": f"❌ Found {len(separated)} points instead of 12. Try better lighting or different angle."}
        #
        # # Label points
        # sorted_values = [v for _, v in sorted(zip(midpoints, pairs), key=lambda t: t[0][1])]
        # labeled_points = dict(zip(LABELS, sorted_values))
        #
        # # Draw lines
        # for c1, c2 in pairs:
        #     cv2.line(output, c1, c2, (0, 255, 0), 2)
        #
        # # Draw circles
        # for pt in separated:
        #     cv2.circle(output, pt, 4, (0, 255, 0), -1)
        #
        # # Draw labels with percentages
        # results_dict = {}
        # for k, v in labeled_points.items():
        #     mid_x, mid_y, l_pcts, r_pcts = compute_percentage(v)
        #
        #     # Add label to image
        #     cv2.putText(output, k, (v[1][0] + 40, v[1][1]),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        #
        #     results_dict[k] = {
        #         "left_pct": f"{l_pcts:.1f}",
        #         "right_pct": f"{r_pcts:.1f}"
        #     }
        #
        # # Draw connection lines
        # try:
        #     sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
        #     dv_1, dv_2 = labeled_points['dimples_of_Venus']
        #     cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
        #     cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)
        # except:
        #     pass
        #
        # output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        #
        # return output_rgb, {
        #     "status": "success",
        #     "count": len(separated),
        #     "data": results_dict
        # }
        image_rgb = np.ascontiguousarray(np.array(image_input), dtype=np.uint8)
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            output_segmentation_masks=True  # Enables segmentation mask equivalent
        )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        with PoseLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(mp_image)

        # 4. Extract segmentation mask
        if results.segmentation_masks:
            # Convert MediaPipe Image mask to NumPy float array
            segmentation_mask = results.segmentation_masks[0].numpy_view()
            mask = (segmentation_mask > 0.7).astype(np.uint8) * 255
        else:
            return None, {"error": "❌ No body detected. Try another photo."}

        foreground = np.where(mask[..., None], image_input, 0)

        h, w, _ = image_rgb.shape
        head_points = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            for idx, lm in enumerate(landmarks):
                px, py = int(lm.x * w), int(lm.y * h)
                head_points.append([px, py])
                if idx >= 10:
                    break
        average_head_point = np.array(head_points).mean(axis=0)

        res = find_blobs(foreground)
        deduplicated_points = deduplicate_points(np.array(res), min_cluster_size=20)
        separated, pairs, midpoints = separate_appropriate_points(deduplicated_points)
        output = image_rgb.copy()


        if len(separated) != 12:
            for pt in separated:
                cv2.circle(output, pt, 7, (0, 255, 0), -1)
            return output, {"error": f'detected {len(separated)} points istead of 12 ...'}

        # label points

        sorted_values = [v for _, v in sorted(zip(midpoints, pairs), key=lambda t: t[0][1])]
        sorted_midpoints = sorted(midpoints, key=lambda t: t[1])

        labeled_points = dict(zip(LABELS, sorted_values))

        # draw a line between the points
        for i in range(len(pairs)):
            c1, c2 = pairs[i]
            cv2.line(output, c1, c2, (0, 255, 0), 2)

        # draw separated points
        for pt in separated:
            cv2.circle(output, pt, 4, (0, 255, 0), -1)

        # draw labels for the points
        for k, v in labeled_points.items():
            mid_x, mid_y, l_pcts, r_pcts = compute_percentage(v)
            cv2.putText(output, k, (v[1][0] + 40, v[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2,
                        cv2.LINE_AA)
            mid_lx, mid_ly = (v[0][0] + mid_x) // 2, (v[0][1] + mid_y) // 2
            mid_rx, mid_ry = (v[1][0] + mid_x) // 2, (v[1][1] + mid_y) // 2

            cv2.putText(output, f"{l_pcts:.2f} %", (mid_lx - 50, mid_ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output, f'{r_pcts:.2f} %', (mid_rx - 50, mid_ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 1, cv2.LINE_AA)

        # connect (draw lines )dimples of Venus and scapula
        sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
        dv_1, dv_2 = labeled_points['dimples_of_Venus']
        cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
        cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)

        # draw a head point
        cv2.circle(output, (int(average_head_point[0]), int(average_head_point[1])), 6, (255, 255, 0), -1)

        return output, {
            "status": "success",
            "count": len(separated),
            "data": dict()
        }

    except Exception as e:
        import traceback
        return None, {"error": f"❌ Error: {str(e)[:100]}"}


# UI
st.markdown("# 🎯 Blob Detection")
st.markdown("Detect anatomical landmarks from photos")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Photo")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input", use_container_width=True)

        if st.button("🔍 Detect Landmarks", use_container_width=True):
            st.session_state.process = True

with col2:
    st.subheader("📊 Results")

    if uploaded and st.session_state.get("process", False):
        with st.spinner("⏳ Processing image (30-60 seconds)..."):
            img = Image.open(uploaded)
            print('image processed', 100 * '-')
            result_img, result_data = process_image(img)

            if result_img is not None:
                st.image(result_img, caption="Detected Landmarks", use_container_width=True)

                # if result_data.get("status") == "success":
                #     st.success(f"✅ Found {result_data['count']} landmarks!")
                #
                #     # # Show data in expander
                #     # with st.expander("📋 Landmark Details"):
                #     #     for label, data in result_data["data"].items():
                #     #         st.write(f"**{label}**  \nLeft: {data['left_pct']}% | Right: {data['right_pct']}%")
                #
                #     # Download
                #     buf = BytesIO()
                #     Image.fromarray(result_img).save(buf, format="JPG")
                #     buf.seek(0)
                #     st.download_button(
                #         "📥 Download Result",
                #         buf,
                #         "blob_detection_result.png",
                #         "image/png",
                #         use_container_width=True
                #     )
                # else:
                #     st.error(result_data.get('error', '❌ Processing failed'))
            else:
                st.error(result_data['error'] + str(mp.__version__))

        st.session_state.process = False

st.divider()
st.markdown("""
### 💡 Tips for best results:
- **Full body visible** from behind
- **Good lighting** (no shadows)
- **Plain background** (white or neutral color)
- **Clear landmarks** visible on skin
- High-quality photo (at least 640x480)

### 📍 Detected Landmarks:
- Knees
- Dimples of Venus
- Angulus Inferior Scapulae
- Margo Lateralis Acromialis
- Epicondyle
- Processus Styloideus
""")
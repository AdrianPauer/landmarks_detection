"""
Streamlit Blob Detection API
Detects anatomical landmarks in body images
"""

import os

os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import torch
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pillow_heif
from mobile_sam import sam_model_registry, SamPredictor
import sys
import math
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
            mask = (segmentation_mask > 0.8).astype(np.uint8) * 255
        else:
            return None, {"error": "❌ No body detected. Try another photo."}

        # identify head point
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

        # identify hip point
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_points = [
            [int(left_hip.x * w), int(left_hip.y * h)],
            [int(right_hip.x * w), int(right_hip.y * h)]
        ]
        hip_point = np.array(hip_points).mean(axis=0)
        hx, hy = int(hip_point[0]), int(hip_point[1])
        point_coords = np.array([[hx, hy]], dtype=np.float32)

        # sam_mask
        model_type = "vit_t"
        checkpoint = "mobile_sam.pt"

        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
        mobile_sam.to(device="cpu")  # Runs smoothly on server CPU

        predictor = SamPredictor(mobile_sam)
        point_labels = np.array([1])

        with torch.inference_mode():
            predictor.set_image(image_rgb)
            masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                      multimask_output=True)
        # choose mask with biggest area
        areas = [mask.sum() for mask in masks]
        mask = masks[np.argmax(areas)].astype(np.uint8) * 255

        foreground = np.where(mask[..., None], image_rgb, 0)

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
            # mid_lx, mid_ly = (v[0][0] + mid_x) // 2, (v[0][1] + mid_y) // 2
            # mid_rx, mid_ry = (v[1][0] + mid_x) // 2, (v[1][1] + mid_y) // 2

            # cv2.putText(output, f"{l_pcts:.2f} %", (mid_lx - 50, mid_ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(output, f'{r_pcts:.2f} %', (mid_rx - 50, mid_ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 1, cv2.LINE_AA)

        # draw a vertical line between midpoints
        for i in range(len(sorted_midpoints)-1):
            mid1,mid2 = sorted_midpoints[i],sorted_midpoints[i+1]
            #cv2.line(output, mid1, mid2, (0, 255, 0), 2)
            cv2.circle(output, mid1, 2, (255, 0, 0), 5)
            cv2.circle(output, mid2, 2, (255, 0, 0), 5)

        # connect (draw lines )dimples of Venus and scapula
        sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
        dv_1, dv_2 = labeled_points['dimples_of_Venus']
        cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
        cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)

        # draw a head point
        cv2.circle(output, (int(average_head_point[0]), int(average_head_point[1])), 6, (255, 255, 0), -1)

        pt1, pt2 = labeled_points['processus_styloideus']
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        dir_x = dx
        dir_y = dy
        line_length = 0.4

        p_start = (int(average_head_point[0] - dir_x * line_length), int(average_head_point[1] - dir_y * line_length))
        p_end = (int(average_head_point[0] + dir_x * line_length), int(average_head_point[1] + dir_y * line_length))
        cv2.line(output, p_start, p_end, (255, 0, 0), 4)

        pt3,pt4 = labeled_points['margo_lateralis_acromialis']
        points = np.array([pt3,p_start,p_end,pt4], dtype=np.int32)  # your 4 points, in order

        poly_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [points], 255)

        head_areas_mask = cv2.bitwise_and(255-mask, poly_mask)
        output[head_areas_mask == 255, 0] = 255
        output[head_areas_mask == 255, 1] = 255
        output[head_areas_mask == 255, 2] = 0

        ys, xs = np.where(head_areas_mask > 0)
        coords = np.column_stack((xs, ys))
        group_right = coords[coords[:, 0] < average_head_point[0]]
        group_left = coords[coords[:, 0] > average_head_point[0]]

        # compute percentages
        if len(group_left) > 0 and len(group_right) > 0:
            gr_pcts = len(group_right) / (len(group_right) + len(group_left)) * 100
            gl_pcts = len(group_left) / (len(group_right) + len(group_left)) * 100

            # Cast mean numpy arrays to python tuples of ints
            right_org = tuple(group_right.mean(axis=0).astype(int))
            left_org = tuple(group_left.mean(axis=0).astype(int))

            cv2.putText(output, f'{gr_pcts:.2f} %', right_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(output, f'{gl_pcts:.2f} %', left_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)

        elif len(group_right) > 0:
            right_org = tuple(group_right.mean(axis=0).astype(int))
            cv2.putText(output, f'{100:.2f} %', right_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        elif len(group_left) > 0:
            left_org = tuple(group_left.mean(axis=0).astype(int))
            cv2.putText(output, f'{100:.2f} %', left_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

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
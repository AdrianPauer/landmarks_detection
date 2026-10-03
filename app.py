"""
Streamlit Blob Detection API
Detects anatomical landmarks in body images
"""

import os
import torch
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pillow_heif
from mobile_sam import sam_model_registry, SamPredictor
import sys
from io import BytesIO
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register HEIF opener
pillow_heif.register_heif_opener()

# Import blob detection functions
from ex3_blob_detection import (
    find_blobs,
    deduplicate_points,
    separate_appropriate_points,
    compute_segment_percentages,
    separate_and_label_dimple_points,
    draw_a_dashed_line,
    line_intersection,
    get_line_mask_intersection,
    calculate_angle_numpy,
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
        error = ''

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

        # Extract segmentation mask
        if results.segmentation_masks is None:
            return None, {"error": "❌ Seems like no human is on the uploaded photo. Try another photo."}

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
        wrist_point = int(landmarks[21].x * w), int(landmarks[21].y * h)

        # identify hip point
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_points = [
            [int(left_hip.x * w), int(left_hip.y * h)],
            [int(right_hip.x * w), int(right_hip.y * h)]
        ]
        hip_point = np.array(hip_points).mean(axis=0)
        hip_x, hip_y  = int(hip_point[0]), int(hip_point[1])
        point_coords = np.array([[hip_x, hip_y]], dtype=np.float32)

        # mask from mobile_sam mask model
        model_type = "vit_t"
        checkpoint = "mobile_sam.pt"

        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
        mobile_sam.to(device="cpu")

        predictor = SamPredictor(mobile_sam)
        point_labels = np.array([1])

        with torch.inference_mode():
            predictor.set_image(image_rgb)
            masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                     multimask_output=True)
        # detect head mask
        hx, hy = int(average_head_point[0]), int(average_head_point[1])
        point_coords = np.array([[hx, hy]], dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(image_rgb)
            head_masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                           multimask_output=False)

        error += 'all masks good'

        # choose mask with biggest area (always mask for the whole body pose)
        areas = [mask.sum() for mask in masks]
        best_mask = masks[np.argmax(areas)].astype(np.uint8) * 255

        # append the head mask to the body mask
        head_mask = head_masks[0]
        new_mask = best_mask.copy()
        new_mask[head_mask] = 0

        # extract only body image as "foreground"
        foreground = np.where(new_mask[..., None], image_rgb, 0)

        detected_markers = find_blobs(foreground)
        deduplicated_points = deduplicate_points(np.array(detected_markers))
        separated, pairs, midpoints, dimple_points = separate_appropriate_points(deduplicated_points,wrist_point,average_head_point, (hip_x, hip_y))
        dimple_point, th_points = separate_and_label_dimple_points(dimple_points)

        # copy original img to output
        output = image_rgb.copy()

        if len(separated) != 14:
            for pt in separated:
                cv2.circle(output, pt, 7, (0, 255, 0), -1)
            return output, {"error": f'detected {len(separated)} points istead of 12 ...'}

        # sort the pairs according to y axis and label from top to bottom
        sorted_values = [v for _, v in sorted(zip(midpoints, pairs), key=lambda t: t[0][1])]
        labeled_points = dict(zip(LABELS, sorted_values))
        labeled_points['dimples_of_Venus'] = dimple_point

        error += ' labled pts '
        # draw a line between the points
        for pair in labeled_points.values():
            c1, c2 = pair
            cv2.line(output, c1, c2, (0, 255, 0), 2)

        # draw labels for the points
        for k, v in labeled_points.items():
            cv2.putText(output, k, (v[1][0] + 60, v[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2,
                        cv2.LINE_AA)

        # connect dimples of Venus and scapula to form trapezium (lichobežník)
        sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
        dv_1, dv_2 = labeled_points['dimples_of_Venus']
        cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
        cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)

        pt1, pt2 = labeled_points['processus_styloideus']
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dir_x = dx
        dir_y = dy
        line_length = 0.4

        p_start = (int(average_head_point[0] - dir_x * line_length), int(average_head_point[1] - dir_y * line_length))
        p_end = (int(average_head_point[0] + dir_x * line_length), int(average_head_point[1] + dir_y * line_length))

        pt3, pt4 = labeled_points['margo_lateralis_acromialis']
        points = np.array([pt3, p_start, p_end, pt4], dtype=np.int32)  # your 4 points, in order

        mask = np.zeros(best_mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        head_areas_mask = cv2.bitwise_and(255 - best_mask, mask)
        output[head_areas_mask == 255] = (0, 255, 255)

        ys, xs = np.where(head_areas_mask > 0)
        coords = np.column_stack((xs, ys))
        group_right = coords[coords[:, 0] < average_head_point[0]]
        group_left = coords[coords[:, 0] > average_head_point[0]]

        # compute percentages
        if len(group_left) > 0 and len(group_right) > 0:
            gr_pcts = len(group_right) / (len(group_right) + len(group_left)) * 100
            gl_pcts = len(group_left) / (len(group_right) + len(group_left)) * 100

            right_pos_x, right_pos_y = group_right[:, 0].mean(), group_right[:, 1].max() + 20
            right_pos = (int(right_pos_x), int(right_pos_y))
            left_pos_x, left_pos_y = group_left[:, 0].mean(), group_left[:, 1].max() + 20
            left_pos = (int(left_pos_x), int(left_pos_y))

            cv2.putText(output, f'{gr_pcts:.1f} %', right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(output, f'{gl_pcts:.1f} %', left_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1,
                        cv2.LINE_AA)
        elif len(group_right) > 0:
            right_pos_x, right_pos_y = group_right[:, 0].mean(), group_right[:, 1].max() + 20
            right_pos = (int(right_pos_x), int(right_pos_y))

            cv2.putText(output, f'{100:.1f} %', right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        elif len(group_left) > 0:
            left_pos_x, left_pos_y = group_left[:, 0].mean(), group_left[:, 1].max() + 20
            left_pos = (int(left_pos_x), int(left_pos_y))

            cv2.putText(output, f'{100:.1f} %', left_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # draw th points and dashed line
        cv2.circle(output, th_points[1], 4, (255, 0, 0), -1)
        cv2.circle(output, th_points[0], 4, (255, 0, 0), -1)
        draw_a_dashed_line(output, th_points[0], th_points[1])

        error += ' head percentages '
        # compute percentages for line intersections at the back
        for key in ['dimples_of_Venus', 'angulus_inferior_scapulae']:
            body_part_pts = labeled_points[key]
            intersection = line_intersection(body_part_pts, th_points)
            compute_segment_percentages(body_part_pts, intersection)
            l_pcts, r_pcts = compute_segment_percentages(body_part_pts, intersection)

            # compute place to draw percentages and draw
            pt1, pt2 = body_part_pts
            mid_lx, mid_ly = (pt1[0] + intersection[0]) // 2, (pt1[1] + intersection[1]) // 2
            mid_rx, mid_ry = (pt2[0] + intersection[0]) // 2, (pt2[1] + intersection[1]) // 2

            cv2.circle(output, intersection, 4, (0, 0, 0), -1)
            cv2.putText(output, f"{l_pcts:.2f} %", (mid_lx - 40, mid_ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1, cv2.LINE_AA)
            cv2.putText(output, f'{r_pcts:.2f} %', (mid_rx - 10, mid_ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1, cv2.LINE_AA)

            # line and mask intersection
            pt1, pt2 = labeled_points['processus_styloideus']
            x1, y1 = pt1
            x2, y2 = pt2
            # Calculate vector P1 -> P2
            dx = (x2 - x1) / math.dist(pt1, pt2)
            dy = (y2 - y1) / math.dist(pt1, pt2)

        for body_part_pt in [labeled_points['epicodyles'][0], labeled_points['epicodyles'][1],
                             labeled_points['processus_styloideus'][0], labeled_points['processus_styloideus'][1]]:
            intersection = body_part_pt
            pt_min_x, pt_max_x = get_line_mask_intersection(new_mask, body_part_pt, (dx, dy))
            cv2.line(output, pt_min_x, pt_max_x, (255, 0, 0), 2)
            cv2.circle(output, pt_min_x, 2, (0, 0, 0), -1)
            cv2.circle(output, pt_max_x, 2, (0, 0, 0), -1)
            cv2.circle(output, intersection, 2, (0, 0, 0), -1)

            l_pcts, r_pcts = compute_segment_percentages((pt_min_x, pt_max_x), intersection)

            pt1, pt2 = pt_min_x, pt_max_x
            mid_lx, mid_ly = (pt1[0] + intersection[0]) // 2, (pt1[1] + intersection[1]) // 2
            mid_rx, mid_ry = (pt2[0] + intersection[0]) // 2, (pt2[1] + intersection[1]) // 2

            cv2.putText(output, f"{l_pcts:.2f} %", (mid_lx - 80, mid_ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(output, f'{r_pcts:.2f} %', (mid_rx - 20, mid_ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 1, cv2.LINE_AA)

            # compute distance arm points -> edge
            for pt, direction in (labeled_points['margo_lateralis_acromialis'][0], 'left'), (
                    labeled_points['margo_lateralis_acromialis'][1], 'right'):
                pt_min_x, pt_max_x = get_line_mask_intersection(new_mask, pt, (dx, dy), direction)
                cv2.line(output, pt_min_x, pt_max_x, (255, 0, 0), 2)
                cv2.circle(output, pt_min_x, 2, (0, 0, 0), -1)
                cv2.circle(output, pt_max_x, 2, (0, 0, 0), -1)
                mid_lx, mid_ly = (pt_min_x[0] + pt_max_x[0]) // 2, (pt_min_x[1] + pt_max_x[1]) // 2
                cv2.putText(output, f"{pt_max_x[0] - pt_min_x[0]} px", (mid_lx - 10, mid_ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)

            # compute the angle arm, elbow, wrist and draw
            for index, (a, b, c) in enumerate(
                    [(labeled_points['processus_styloideus'][0], labeled_points['epicodyles'][0],
                      labeled_points['margo_lateralis_acromialis'][0]),
                     (labeled_points['margo_lateralis_acromialis'][1], labeled_points['epicodyles'][1],
                      labeled_points['processus_styloideus'][1])]):

                angle = calculate_angle_numpy(a, b, c)
                if index == 1:
                    angle = 360 - angle
                    ba = np.array(b) - np.array(a)
                    bc = np.array(b) - np.array(c)
                    text_pos = (b[0] - 70, b[1] + 30)
                else:
                    ba = np.array(a) - np.array(b)
                    bc = np.array(c) - np.array(b)
                    text_pos = (b[0] + 10, b[1] + 30)

                draw_a_dashed_line(output, a, b, color=(0, 255, 255))
                draw_a_dashed_line(output, b, c, color=(0, 255, 255))

                start_angle = np.degrees(np.arctan2(ba[1], ba[0]))
                end_angle = np.degrees(np.arctan2(bc[1], bc[0]))

                if abs(end_angle - start_angle) > 180:
                    if start_angle < end_angle:
                        start_angle += 360
                    else:
                        end_angle += 360

                cv2.ellipse(output, center=b, axes=(30, 30), angle=0, startAngle=min(start_angle, end_angle),endAngle=max(start_angle, end_angle), color=(0, 255, 255), thickness=1,lineType=cv2.LINE_AA)
                cv2.putText(output, str(angle) + '°', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1,cv2.LINE_AA)

        return output, {
            "status": "success",
            "count": len(separated),
            "data": dict()
        }

    except Exception as e:
        return None, {"error": f"❌ Error: {str(e)[:100]}"}


# UI
st.markdown("# 🎯 Blob Detection")
st.markdown("Detect anatomical landmarks from photos")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Photo")
    uploaded = st.file_uploader("Choose image", type=['jpg','HEIC','heic','heif'])

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

                if result_data.get("status") == "success":
                    st.success(f"✅ Found {result_data['count']} landmarks!")


                    # Download
                    buf = BytesIO()
                    Image.fromarray(result_img).save(buf, format="JPEG")
                    buf.seek(0)
                    st.download_button(
                        "📥 Download Result",
                        buf,
                        "blob_detection_result.png",
                        "image/png",
                        use_container_width=True
                    )
                else:
                    st.error(result_data.get('error', '❌ Processing failed'))
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
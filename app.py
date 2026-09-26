"""
Streamlit API for Blob Detection
This wraps the blob detection script for easy use via web interface and Hugging Face Spaces
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pillow_heif
import torch
import math
import json
from io import BytesIO

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
    page_title="Blob Detection API",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAM2_AVAILABLE = False
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    pass


def process_image(image_input):
    """
    Process image and detect blob landmarks.
    
    Args:
        image_input: PIL Image
        
    Returns:
        tuple: (output_image, results_dict)
    """
    try:
        # Convert PIL Image to numpy array
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Convert to RGB for mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize mediapipe pose detector
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=2,
            enable_segmentation=True
        ) as pose:
            results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, {"error": "No pose detected in image"}
        
        # Segmentation with SAM2 if available
        if SAM2_AVAILABLE:
            try:
                checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                device = "cpu"
                
                model = build_sam2(model_cfg, checkpoint, device=device)
                predictor = SAM2ImagePredictor(model)
                
                landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                point_coords = np.array([[int(landmark.x * w), int(landmark.y * h) + 200]])
                point_labels = np.array([1])
                
                with torch.inference_mode():
                    predictor.set_image(image_rgb)
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                
                # Choose mask with biggest area
                areas = [mask.sum() for mask in masks]
                best_mask = masks[np.argmax(areas)].astype(np.uint8) * 255
            except Exception as e:
                if results.segmentation_mask is not None:
                    best_mask = (results.segmentation_mask > 0.8).astype(np.uint8) * 255
                else:
                    best_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            if results.segmentation_mask is not None:
                best_mask = (results.segmentation_mask > 0.8).astype(np.uint8) * 255
            else:
                best_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Create foreground
        foreground = np.where(best_mask[..., None], image, 0)
        
        # Get head points for reference
        head_points = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for idx, lm in enumerate(landmarks):
                px, py = int(lm.x * w), int(lm.y * h)
                head_points.append([px, py])
                if idx >= 10:
                    break
        
        average_head_point = np.array(head_points).mean(axis=0) if head_points else np.array([w // 2, h // 2])
        
        # Find blob landmarks
        res = find_blobs(foreground)
        deduplicated_points = deduplicate_points(np.array(res), min_cluster_size=20)
        separated, pairs, midpoints = separate_appropriate_points(deduplicated_points)
        
        # Prepare output
        output = image.copy()
        
        # Check if correct number of landmarks detected
        if len(separated) != 12:
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), {"error": f"Detected {len(separated)} points instead of 12", "landmarks": []}
        
        # Label the points
        sorted_values = [v for _, v in sorted(zip(midpoints, pairs), key=lambda t: t[0][1])]
        sorted_midpoints = sorted(midpoints, key=lambda t: t[1])
        labeled_points = dict(zip(LABELS, sorted_values))
        
        # Draw lines between landmark pairs
        for i in range(len(pairs)):
            c1, c2 = pairs[i]
            cv2.line(output, c1, c2, (0, 255, 0), 2)
        
        # Draw landmark circles
        for pt in separated:
            cv2.circle(output, pt, 4, (0, 255, 0), -1)
        
        # Draw labels and percentages
        results_dict = {}
        for k, v in labeled_points.items():
            mid_x, mid_y, l_pcts, r_pcts = compute_percentage(v)
            cv2.putText(
                output, k, (v[1][0] + 40, v[1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2,
                cv2.LINE_AA
            )
            mid_lx, mid_ly = (v[0][0] + mid_x) // 2, (v[0][1] + mid_y) // 2
            mid_rx, mid_ry = (v[1][0] + mid_x) // 2, (v[1][1] + mid_y) // 2
            
            cv2.putText(
                output, f"{l_pcts:.2f} %", (mid_lx - 50, mid_ly + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                cv2.LINE_AA
            )
            cv2.putText(
                output, f'{r_pcts:.2f} %', (mid_rx - 50, mid_ry + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                cv2.LINE_AA
            )
            
            results_dict[k] = {
                "coordinates": [list(v[0]), list(v[1])],
                "left_percentage": float(f"{l_pcts:.2f}"),
                "right_percentage": float(f"{r_pcts:.2f}")
            }
        
        # Draw connection lines for dimples_of_Venus and scapula
        sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
        dv_1, dv_2 = labeled_points['dimples_of_Venus']
        cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
        cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)
        
        # Head area separation
        pt1, pt2 = labeled_points['processus_styloideus']
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        line_length = 0.4
        
        p_start = (
            int(average_head_point[0] - dx * line_length),
            int(average_head_point[1] - dy * line_length)
        )
        p_end = (
            int(average_head_point[0] + dx * line_length),
            int(average_head_point[1] + dy * line_length)
        )
        cv2.line(output, p_start, p_end, (255, 0, 0), 4)
        
        pt3, pt4 = labeled_points['margo_lateralis_acromialis']
        points = np.array([pt3, p_start, p_end, pt4], dtype=np.int32)
        
        mask = np.zeros(best_mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        head_areas_mask = cv2.bitwise_and(255 - best_mask, mask)
        output[head_areas_mask == 255, 0] = 255
        output[head_areas_mask == 255, 1] = 255
        output[head_areas_mask == 255, 2] = 0
        
        # Compute head side percentages
        ys, xs = np.where(head_areas_mask > 0)
        coords = np.column_stack((xs, ys))
        group_right = coords[coords[:, 0] < average_head_point[0]]
        group_left = coords[coords[:, 0] > average_head_point[0]]
        
        head_percentages = {}
        if len(group_left) > 0 and len(group_right) > 0:
            gr_pcts = len(group_right) / (len(group_right) + len(group_left)) * 100
            gl_pcts = len(group_left) / (len(group_right) + len(group_left)) * 100
            head_percentages = {"right": float(f"{gr_pcts:.2f}"), "left": float(f"{gl_pcts:.2f}")}
        elif len(group_right) > 0:
            head_percentages = {"right": 100.0}
        elif len(group_left) > 0:
            head_percentages = {"left": 100.0}
        
        cv2.circle(
            output,
            (int(average_head_point[0]), int(average_head_point[1])),
            6, (255, 255, 0), -1
        )
        
        # Convert back to RGB for display
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        return output_rgb, {
            "status": "success",
            "landmarks_detected": len(separated),
            "landmarks": results_dict,
            "head_percentages": head_percentages
        }
        
    except Exception as e:
        return None, {"error": str(e), "landmarks": []}


# Streamlit interface
st.markdown("""
# 🎯 Blob Detection API
Detect and label anatomical landmarks in body images using AI.

**Supported Landmarks:**
- Knees
- Dimples of Venus
- Angulus Inferior Scapulae (lower shoulder blade)
- Margo Lateralis Acromialis (shoulder edge)
- Epicondyle (elbow)
- Processus Styloideus (wrist)
""")

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a clear photo from behind
    2. Ensure full body is visible
    3. Click 'Detect Landmarks'
    4. View results and download annotated image
    """)
    st.markdown("---")
    st.markdown("**Best Results With:**")
    st.markdown("✓ Plain background\n✓ Full body in frame\n✓ Clear landmarks visible")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'heic', 'heif'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### Results")
    if uploaded_file is not None:
        if st.button("🔍 Detect Landmarks", key="detect_btn"):
            with st.spinner("Processing image..."):
                output_image, results = process_image(image)
                
                if output_image is not None:
                    st.image(output_image, caption="Annotated Result", use_column_width=True)
                    
                    # Display results
                    if results.get("status") == "success":
                        st.success(f"✅ Found {results['landmarks_detected']} landmarks!")
                        
                        # Landmarks table
                        st.markdown("#### Landmark Details")
                        for label, data in results["landmarks"].items():
                            with st.expander(f"📍 {label}"):
                                st.write(f"**Left %:** {data['left_percentage']}%")
                                st.write(f"**Right %:** {data['right_percentage']}%")
                                st.write(f"**Points:** {data['coordinates']}")
                        
                        # Head percentages
                        if results.get("head_percentages"):
                            st.markdown("#### Head Area Distribution")
                            head_data = results["head_percentages"]
                            col_a, col_b = st.columns(2)
                            if "left" in head_data:
                                col_a.metric("Left Side", f"{head_data['left']:.1f}%")
                            if "right" in head_data:
                                col_b.metric("Right Side", f"{head_data['right']:.1f}%")
                        
                        # Download button
                        buf = BytesIO()
                        Image.fromarray(output_image).save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Result",
                            data=buf,
                            file_name="blob_detection_result.png",
                            mime="image/png"
                        )
                    else:
                        st.error(f"❌ {results.get('error', 'Unknown error')}")
                else:
                    st.error("Failed to process image")

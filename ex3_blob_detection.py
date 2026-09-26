import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pillow_heif
import argparse
import torch
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



LABELS = ['knees','dimples_of_Venus','angulus_inferior_scapulae','margo_lateralis_acromialis','epicodyle','processus_styloideus']
def find_blobs(hsv_image):
    hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
    saturations_values = [30,40,60,80,100,120]
    min_saturation_values =[160,170,180,190,200,210]
    detected_blobs = []
    filters = {'arms' : [np.array([0, 100, 1]),np.array([5, 255, 255]), 0.4],
               'scapulas': [np.array([0, 0, 1]),  np.array([10, 255, 255]), 0.4],
               'other': [np.array([170, 100, 1]), np.array([179, 255, 255]), 0.3]
               }
    for saturation in saturations_values:
        # original filter
        lower_white = np.array([0, 0, 1])
        upper_white = np.array([360, saturation, 255])
        # orange filter
        # lower_white = np.array([5, saturation, 80])
        # upper_white = np.array([200, 255, 255])

        # arms
        # lower_red1 = np.array([0, 100, 1])
        # upper_red1 = np.array([5, 255, 255])

        # scapulas
        # lower_red1 = np.array([0, 0, 1])
        # upper_red1 = np.array([10, 255, 255])

        #most of them
        # lower_red1 = np.array([170, 100, 1])
        # upper_red1 = np.array([179, 255, 255])

        white_mask = cv2.inRange(hsv,lower_white,upper_white)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(args.image_path[:-5] + f"{saturation}_analyzed.jpg", white_mask)


        radius = 12
        template_size = radius * 2 + 4
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(template, (template_size // 2, template_size // 2), radius, 255, -1)  # filled circle

        result = cv2.matchTemplate(white_mask, template, cv2.TM_CCOEFF_NORMED)

        # Find all locations above a match-quality threshold
        threshold = 0.6
        locations = np.where(result >= threshold)
        template_h, template_w = template.shape  # or template.shape[:2] if 3-channel
        points = list(zip(*locations[::-1]))  # (x, y) pairs
        points = [(x + template_w // 2, y + template_h // 2) for x, y in points]
        for p in points:
            detected_blobs.append(p)
    return detected_blobs


def deduplicate_points(points, min_distance=20, min_cluster_size=2):
    """
    Collapse clusters of nearby points into one representative point per cluster.
    Only keeps clusters with at least min_cluster_size points.
    points: list of (x, y)
    """
    if len(points) == 0:
        return []

    points = np.array(points)
    kept = []
    used = np.zeros(len(points), dtype=bool)
    cluster_sizes = []
    for i in range(len(points)):
        if used[i]:
            continue

        dists = np.hypot(points[:, 0] - points[i][0], points[:, 1] - points[i][1])
        cluster_mask = dists < min_distance
        cluster_points = points[cluster_mask]
        cluster_size = len(cluster_points)

        used[cluster_mask] = True


        avg_point = tuple(cluster_points.mean(axis=0).astype(int))
        kept.append((avg_point,cluster_size))
        cluster_sizes.append(cluster_size)

    sorted_kept = sorted(kept, key=lambda x: x[1])
    gaps = [sorted_kept[i+1][1] - sorted_kept[i][1] for i in range(len(cluster_sizes) - 1)]
    signif_gap_idx = np.where(np.array(gaps) > 25)[0][0].item()
    new_kept =[x[0] for x in sorted_kept[signif_gap_idx + 1:]]
    return new_kept

def separate_appropriate_points(points):
    separated, pairs, midpoints= [],[],[]
    for c1 in points:
        x1,y1 = c1
        for c2 in points:
            if c1 == c2 or (c2,c1) in pairs or (c1,c2) in pairs: continue
            x2,y2 = c2
            if abs(y1 - y2) < 50 and 50 < abs(x1 - x2) < 1500 :
                separated.append(c1)
                separated.append(c2)
                # sort according to x-coordinate
                if c1[0] < c2[0] :
                    pairs.append((c1,c2))
                else:
                    pairs.append((c2, c1))
                mid_x = (c1[0] + c2[0]) // 2
                mid_y = (c1[1] + c2[1]) // 2
                midpoints.append((mid_x, mid_y))
                break
    return separated,pairs,midpoints

def compute_percentage(pair):
    x,y = pair[0], pair[1]
    mid_x = (x[0] + y[0]) / 2
    mid_y = (x[1] + y[1]) / 2
    lenght = ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** (1/2)
    left_len = ((mid_x - x[0]) ** 2 + (mid_y - x[1]) ** 2) **(1/2)
    left_len_pcts = left_len / lenght
    right_len_pcts = 1 - left_len_pcts
    return int(mid_x), int(mid_y), left_len_pcts, right_len_pcts



if __name__ == "__main__":
    # load and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    # transform data from .HEIF format to .jpg
    #imgs = ["straight_after.HEIC", "straight_before.HEIC", "straight_side.HEIC", "straight_none.HEIC", "tuck_after.heic","tuck_before.HEIC","tuck_side.heic","tuck_none.HEIC"]
    pillow_heif.register_heif_opener()
    img = Image.open(args.image_path)

    img.save(args.image_path[:-4] + "jpg")

    # set up the media pip model for segmentation and pose landmark detection
    mp_selfie = mp.solutions.selfie_segmentation
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # load the image
    image = cv2.imread(args.image_path[:-4] + "jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # segmentation mask
    # with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
    #     results = segmenter.process(image_rgb)
    #
    # mask = results.segmentation_mask > 0.6
    # foreground = np.where(mask[..., None], image, 0)

    # pose landmark detection and head points extraction
    h, w, _ = image.shape
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,model_complexity=2,enable_segmentation=True) as pose:
        results = pose.process(image_rgb)

    # if results.segmentation_mask is not None:
    #     body_mask = (results.segmentation_mask > 0.8).astype(np.uint8) * 255

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cpu"

    model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    point_coords = np.array([[int(landmark.x * w), int(landmark.y * h) +200]])
    point_labels = np.array([1])

    with torch.inference_mode():
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                  multimask_output=True)
    # choose mask with biggest area
    areas = [mask.sum() for mask in masks]
    best_mask = masks[np.argmax(areas)].astype(np.uint8) * 255
    overlay = np.zeros_like(image)
    overlay[best_mask > 0] = [0, 255, 0]  # green where mask is present

    foreground = np.where(best_mask[..., None], image, 0)

    head_points = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for idx, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            head_points.append([px, py])
            if idx >= 10:
                break
    average_head_point = np.array(head_points).mean(axis=0)

    # find all the white circle marks in the image
    res = find_blobs(foreground)
    deduplicated_points = deduplicate_points(np.array(res), min_cluster_size=20)
    separated, pairs, midpoints = separate_appropriate_points(deduplicated_points)
    output = image.copy()

    # define labels and label the points according to it
    sorted_values = [v for _, v in sorted(zip(midpoints, pairs), key=lambda t: t[0][1])]
    sorted_midpoints =  sorted(midpoints, key=lambda t: t[1])
    assert len(separated) == 12 , f'detected {len(separated)} points istead of 12 ...'
    labeled_points = dict(zip(LABELS,sorted_values))

    # draw a line between the points
    for i in range(len(pairs)):
        c1, c2 = pairs[i]
        cv2.line(output, c1, c2, (0, 255, 0), 2)

    # draw a vertical line between midpoints
    # for i in range(len(sorted_midpoints)-1):
    #     mid1,mid2 = sorted_midpoints[i],sorted_midpoints[i+1]
    #     cv2.line(output, mid1, mid2, (0, 255, 0), 2)
    #     cv2.circle(output, mid1, 2, (0, 0, 255), 10)
    #     cv2.circle(output, mid2, 2, (0, 0, 255), 10)

    # draw separated points
    for pt in separated:
        cv2.circle(output, pt, 4, (0, 255, 0), -1)

    # draw labels for the points
    for k, v in labeled_points.items():
        mid_x, mid_y, l_pcts, r_pcts = compute_percentage(v)
        cv2.putText(output, k, (v[1][0] + 40, v[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2, cv2.LINE_AA)
        mid_lx,mid_ly = (v[0][0] + mid_x) // 2, (v[0][1] + mid_y) // 2
        mid_rx,mid_ry = (v[1][0] + mid_x) // 2, (v[1][1] + mid_y) // 2

        cv2.putText(output,f"{l_pcts:.2f} %", (mid_lx - 50, mid_ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(output, f'{r_pcts:.2f} %', (mid_rx - 50, mid_ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)

    # connect (draw lines )dimples of Venus and scapula
    sc_1, sc_2 = labeled_points['angulus_inferior_scapulae']
    dv_1, dv_2 = labeled_points['dimples_of_Venus']
    cv2.line(output, sc_1, dv_1, (0, 255, 0), 2)
    cv2.line(output, sc_2, dv_2, (0, 255, 0), 2)


    # separate surfaces around the head
    pt1,pt2 = labeled_points['processus_styloideus']
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

    mask = np.zeros(best_mask.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    head_areas_mask = cv2.bitwise_and(255-best_mask, mask)
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
        cv2.putText(output, f'{gr_pcts:.2f} %', group_right.mean(axis=0).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),1, cv2.LINE_AA)
        cv2.putText(output, f'{gl_pcts:.2f} %', group_left.mean(axis=0).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 1,cv2.LINE_AA)
    elif len(group_right) > 0 :
        cv2.putText(output, f'{100:.2f} %', group_right.mean(axis=0).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)
    elif len(group_left) > 0 :
        cv2.putText(output, f'{100:.2f} %', group_right.mean(axis=0).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)


    cv2.circle(output, (int(average_head_point[0]), int(average_head_point[1])), 6, (255, 255, 0), -1)
    cv2.imwrite(args.image_path[:-5] + "_analyzed.jpg", output)
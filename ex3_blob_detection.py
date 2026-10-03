import cv2
import mediapipe as mp
import argparse
import torch
import math
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import os

LABELS = ['heels','knees','angulus_inferior_scapulae','margo_lateralis_acromialis','epicodyles','processus_styloideus']

def find_blobs(hsv_image):
    hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
    saturations_values = [20,30,40,60,80,100,120]
    detected_blobs = []
    for saturation in saturations_values:
        # white color filter
        lower_white = np.array([0, 0, 1])
        upper_white = np.array([360, saturation, 255])

        white_mask = cv2.inRange(hsv,lower_white,upper_white)
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # define circular template (kernel)
        radius = 7
        template_size = radius * 2 + 4
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(template, (template_size // 2, template_size // 2), radius, 255, -1)

        result = cv2.matchTemplate(white_mask, template, cv2.TM_CCOEFF_NORMED)

        # Find all locations above a match-quality threshold
        threshold = 0.6
        locations = np.where(result >= threshold)
        template_h, template_w = template.shape
        points = list(zip(*locations[::-1]))
        points = [(x + template_w // 2, y + template_h // 2) for x, y in points]
        for p in points:
            detected_blobs.append(p)
    return detected_blobs


def deduplicate_points(points, min_distance=20, min_cluster_size=2):
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

    # sort according to cluster size
    sorted_kept = sorted(kept, key=lambda x: x[1])
    gaps = [sorted_kept[i+1][1] - sorted_kept[i][1] for i in range(len(cluster_sizes) - 1)]

    # find a first big gap in the cluster sizes keep only these points or 20 pts with biggest cluster size
    signif_gap_idx = np.where(np.array(gaps) > 25)[0][0].item()
    new_kept =[x[0] for x in sorted_kept[signif_gap_idx + 1:]]
    if len(kept) - (signif_gap_idx + 1) < 20:
        new_kept = [x[0] for x in sorted_kept[-20:]]

    return new_kept

def separate_appropriate_points(points, wrist_point, head_point, hip_point):
    wx,wy = wrist_point
    hx,hy = hip_point
    head_x,head_y = head_point
    dimple_points = []
    separated, pairs, midpoints= [],[],[]

    # separate dimple points
    for c1 in points:
        x1,y1 = c1
        if y1 > head_y or y1 < hy : continue
        if abs(x1 - head_x) < 70:
            dimple_points.append(c1)
            separated.append(c1)

    # separate the pairs of points with cca same y coordinates
    for c1 in points:
        if c1 in separated:
            continue
        x1,y1 = c1
        for c2 in points:
            if c1 == c2 or (c2,c1) in pairs or (c1,c2) in pairs: continue
            x2,y2 = c2
            # if the points are under the wrist point skip it
            if y1 > wy or y2 > wy : continue
            if abs(y1 - y2) < 50 and abs(x1 - x2) < 1500 :
                separated.append(c1)
                separated.append(c2)
                # sort according to x-coordinate
                if c1[0] < c2[0] :
                    pairs.append((c1,c2))
                else:
                    pairs.append((c2, c1))

                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                midpoints.append((mid_x, mid_y))
                break

    return separated,pairs,midpoints, dimple_points


def compute_segment_percentages(line, p_int):
    """
    Computes the percentage of a segment P1->P2 divided by P_int.
    Assumes P_int lies on the segment P1->P2.
    """
    p1,p2 = line
    total_len = math.dist(p1, p2)

    if total_len == 0:
        return 0.0, 0.0

    left_len = math.dist(p1, p_int)
    left_pct = (left_len / total_len) * 100.0
    right_pct = 100.0 - left_pct

    return round(left_pct, 2), round(right_pct, 2)

def separate_and_label_dimple_points(dimple_points):
    if len(dimple_points) != 4:
        print('wrong number of dimple points')
        return None

    dimple_points = np.array(dimple_points)
    sorted_by_y = dimple_points[dimple_points[:, 1].argsort()]
    th_point = sorted_by_y[-1]
    sorted_by_x = sorted_by_y[sorted_by_y[:-1, 0].argsort()]
    dimpe_pair = (sorted_by_x[0], sorted_by_x[2])
    th_pair = (sorted_by_x[1],th_point)
    return dimpe_pair, th_pair

def draw_a_dashed_line(img, pt1, pt2, color= (255,0,0)):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    # Calculate total line distance
    dist = np.linalg.norm(pt2 - pt1)
    if dist == 0:
        return img

    # Unit vector pointing from pt1 to pt2
    unit_vec = (pt2 - pt1) / dist

    # Draw segments along the line
    current_dist = 0.0
    dash_length = 20
    while current_dist < dist:
        start_dash = pt1 + unit_vec * current_dist
        end_dash = pt1 + unit_vec * min(current_dist + dash_length, dist)
        cv2.line(img, tuple(start_dash.astype(int)), tuple(end_dash.astype(int)),color,1)
        current_dist += dash_length + 5

    return img


def line_intersection(line1, line2):
    """
    Finds the intersection point of two infinite lines, based on determinants.

    :param line1: Tuple of two points ((x1, y1), (x2, y2))
    :param line2: Tuple of two points ((x3, y3), (x4, y4))
    :return: (x, y) point tuple if intersecting, None if parallel
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Lines are parallel or collinear
    if denom == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(round(px)), int(round(py))

def get_line_mask_intersection(mask, point, vertical_vector, direction = 'both'):
    line_mask = np.zeros_like(mask)
    x0,y0 = point
    dx,dy = vertical_vector
    length = 50
    x_start = int(round(x0 - length * dx))
    y_start = int(round(y0 - length * dy))

    x_end = int(round(x0 + length * dx))
    y_end = int(round(y0 + length * dy))

    if direction == 'both':
        p1_new = (x_start, y_start)
        p2_new = (x_end, y_end)
    elif direction == 'left':
        p1_new = (x_start, y_start)
        p2_new = (x0, y0)
    elif direction == 'right':
        p1_new = (x0, y0)
        p2_new = (x_end, y_end)

    cv2.line(line_mask, p1_new, p2_new, (255, 0, 0), 2)
    intersected_mask = cv2.bitwise_and(line_mask, mask)

    ys, xs = np.where(intersected_mask > 0)
    coords = np.column_stack((xs, ys))

    xs = coords[:, 0]
    min_x_idx = np.argmin(xs)
    max_x_idx = np.argmax(xs)
    pt_min_x = tuple(coords[min_x_idx])
    pt_max_x = tuple(coords[max_x_idx])
    return pt_min_x, pt_max_x

def calculate_angle_numpy(a, b, c):
    ba = (np.array(a) - np.array(b))
    bc =( np.array(c) - np.array(b))
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clip to prevent domain errors due to floating-point precision
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle),1)
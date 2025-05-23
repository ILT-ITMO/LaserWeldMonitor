import cv2
import numpy as np

import scipy
import scipy.spatial
from scipy.signal import savgol_filter
import sympy.geometry as gm

import heapq

def approximte_contour(cnt: np.array, epsilon=0.04):
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return approx



def n_max_contours(mask, n=1):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return  sorted(contours, key=cv2.contourArea)[:n]

def bounding_rectangle(cnt):
    bb = cv2.minAreaRect(cnt)
    return np.int0(cv2.boxPoints(bb))

def plate_width_line(rects, main_object_con):
    if (len(rects)==1):
        main_object_con = np.squeeze(main_object_con[0])

        # Sort points by x-coordinate (left-right)
        sorted_points = main_object_con[np.argsort(main_object_con[:, 0])]

        # Split the points into left and right parts based on x-coordinate
        # Left side: points with smaller x values
        middle_index = len(sorted_points) // 2
        
        # Left half: points in the first half of sorted points
        left_part = sorted_points[:middle_index]
        # print("left",left_part)
        
        # Right half: points in the second half of sorted points
        right_part = sorted_points[middle_index:]
        right_part = right_part[::-1]
        # print("right",right_part)

        # Get top 10% points from each side
        percentage = 0.2
        left_part_top = left_part[:int(len(left_part) * percentage)]
        
         
        
        right_part_top = right_part[:int(len(right_part) * percentage)]
        
        # Sort the left and right parts by y-coordinate to get the topmost points
        left_part_top_sorted = left_part_top[np.argsort(left_part_top[:, 1])]
        right_part_top_sorted = right_part_top[np.argsort(right_part_top[:, 1])]

        # Get the topmost point from the left side (top-left)
        top_left = left_part_top_sorted[0]  # The first point in sorted (smallest y)

        # Get the topmost point from the right side (top-right)
        top_right = right_part_top_sorted[0]  # The first point in sorted (smallest y)

        
        rect = np.squeeze(rects[0])

        # Sort points by y-coordinate (to find top & bottom)
        sorted_points = rect[np.argsort(rect[:, 1])]

        # Top line: Get two closest points at the highest part
        percentage=0.5
        # print("sorrytired points",sorted_points)
        # top_part = sorted_points[:int(len(sorted_points) * percentage)]
        # print("top", top_part)
        # top_left = top_part[np.argmin(top_part[:, 1])]
        # top_right = top_part[np.argmax(top_part[:, 1])]
        # line1 = (top_left,  top_right)

        # Get the bottom X% of points
        bottom_part = sorted_points[-int(len(sorted_points) * percentage):]
    
        # Find the leftmost and rightmost points
        bottom_left = bottom_part[np.argmin(bottom_part[:, 0])]
        bottom_right = bottom_part[np.argmax(bottom_part[:, 0])]
        line1 = (top_right,  bottom_right)
        line2 = (top_left, bottom_left)

        # dists = scipy.spatial.distance.pdist(bottom_points)
        # p1_idx, p2_idx = np.unravel_index(np.argmax(dists), (len(bottom_points), len(bottom_points)))
        # line2 = (bottom_points[p1_idx], bottom_points[p2_idx])
       

        return line1, line2
    if (len(rects)==2):
        rect1= rects[0]
        rect2=rects[1]
        rect1, rect2 = np.squeeze(rect1), np.squeeze(rect2)
        d = scipy.spatial.distance.cdist(rect1, rect2)
        (p11_idx, p21_idx), (p12_idx, p22_idx) = heapq.nsmallest(2, np.ndindex(d.shape), key=d.__getitem__)
        
        line1 = rect1[p11_idx], rect1[p12_idx]
        line2 = rect2[p21_idx], rect2[p22_idx]
        return line1, line2
def get_remaining_sides(rect1, rect2):
    """
    Given two rectangles (Nx2 arrays of 4 points each), return the remaining
    sides (line3, line4) that are not the closest pair.

    Each rect is assumed to have shape (4, 2), and points are ordered.
    """
    rect1 = np.squeeze(rect1)
    rect2 = np.squeeze(rect2)

    # Compute pairwise distances
    dists = scipy.spatial.distance.cdist(rect1, rect2)

    # Find the closest two pairs of points
    (p11_idx, p21_idx), (p12_idx, p22_idx) = heapq.nsmallest(2, np.ndindex(dists.shape), key=dists.__getitem__)

    # Middle lines (already returned in your original code)
    middle1 = rect1[p11_idx], rect1[p12_idx]
    middle2 = rect2[p21_idx], rect2[p22_idx]

    # Find remaining indices
    used1 = {p11_idx, p12_idx}
    used2 = {p21_idx, p22_idx}
    remaining1 = [pt for i, pt in enumerate(rect1) if i not in used1]
    remaining2 = [pt for i, pt in enumerate(rect2) if i not in used2]

    # Assume the two remaining points form one of the other sides
    line3 = remaining1[0], remaining1[1]
    line4 = remaining2[0], remaining2[1]

    return line3, line4
# def plate_width_line(rect1, rect2):
#     rect1, rect2 = np.squeeze(rect1), np.squeeze(rect2)
#     d = scipy.spatial.distance.cdist(rect1, rect2)
#     (p11_idx, p21_idx), (p12_idx, p22_idx) = heapq.nsmallest(2, np.ndindex(d.shape), key=d.__getitem__)
    
#     line1 = rect1[p11_idx], rect1[p12_idx]
#     line2 = rect2[p21_idx], rect2[p22_idx]
#     return line1, line2

def calculate_projection(line, p):
    a = line[1] - line[0]
    a = a / np.linalg.norm(a)
    p1 = p - line[0]
    proj = np.dot(p1, a) * a + line[0]
    return proj

def calculate_bias(line1, line2):
    p11, p12 = line1
    p21, p22 = line2
    
    proj = calculate_projection(line2, p11)
    bias = np.linalg.norm(proj - p21)
    return bias
import numpy as np

def compute_misalignment(p1, p2, ref_line):
    """
    Calculate the misalignment (vertical deviation) between a line (p1 to p2)
    and the direction perpendicular to ref_line.

    Parameters:
        p1 (tuple or list): First point of the yellow line (e.g., p11 or p12)
        p2 (tuple or list): Second point of the yellow line (e.g., p21 or p22)
        ref_line (tuple of tuples): A line represented by two points, e.g., ((x1, y1), (x2, y2))

    Returns:
        float: The misalignment (in pixels)
    """

    # Convert to numpy arrays
    v_yellow = np.array(p2) - np.array(p1)
    v_main = np.array(ref_line[1]) - np.array(ref_line[0])

    # Perpendicular vector to the main side
    v_perp = np.array([-v_main[1], v_main[0]])

    # Normalize vectors
    v_yellow_norm = v_yellow / np.linalg.norm(v_yellow)
    v_perp_norm = v_perp / np.linalg.norm(v_perp)

    # Compute angle in radians between yellow line and ideal vertical
    dot_product = np.dot(v_yellow_norm, v_perp_norm)
    theta = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Length of yellow line
    L = np.linalg.norm(v_yellow)

    # Misalignment
    misalignment = L * np.sin(theta)

    return misalignment

def line_intersection_contur(lines, contour):
    contour = np.squeeze(contour)
    m = contour.shape[0]
    results = []
    for line in lines:
        d = scipy.spatial.distance.cdist(contour, line)
        i1, i2 = np.argmin(d, axis=0)
        c1 = np.take(contour, np.arange(i1, i2 + m * (i1 > i2) + 1), mode='wrap', axis=0)
        c2 = np.take(contour, np.arange(i2, i1 + m * (i2 > i1) + 1), mode='wrap', axis=0)

        c1 = np.expand_dims(c1, axis=1)
        c2 = np.expand_dims(c2, axis=1)

        results.append(min([c1, c2], key=lambda x: cv2.arcLength(x, closed=False)))
    return  results


def contour_mass_center(cnt):
    m = cv2.moments(cnt)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    cx, cy

def find_deviation_peaks(line, cnt, threshold):
    cnt = np.squeeze(cnt)
    p1, p2 = line
    if p2[0] < p1[0]:
        p1, p2 = p2, p1
    a = p2 - p1
    a = a / np.linalg.norm(a) #direction
    projections = np.stack([np.dot(cnt - p1, a)] * 2, axis=-1) * np.stack([a] * cnt.shape[0])
    d = np.linalg.norm(cnt - p1 - projections, axis=-1)
    dets  = -np.linalg.det(np.stack([np.stack([a] * cnt.shape[0], axis=0), cnt - p1 - projections], axis=-1))
    dets = np.sign(dets)
    # window_size = max(d.shape[0] // 5, 1)
    # d = savgol_filter(d, window_size, window_size - 1)
    d = np.multiply(dets, d)
    idx = [np.argmax(d), np.argmin(d)]
    dist = d[idx]
    cnts = cnt[idx]
    projs = (projections[idx] + p1).astype(np.uint32)
    return dist, idx, cnts, projs


def calculate_projection_line_width(line1, line2):
    p1 = calculate_projection(line1, line2[0])
    p2 = calculate_projection(line1, line2[1])

    return np.linalg.norm(p1 - p2)
def perpendicular_foot(A, B, C):
    # """Finds the perpendicular foot of point C onto line AB."""
    # A, B, C = map(np.array, (A, B, C))  # Convert to NumPy arrays
    # AB = B - A
    # AC = C - A
    # proj_length = np.dot(AC, AB) / np.dot(AB, AB)  # Projection scalar
    # P = A + proj_length * AB  # Compute perpendicular foot
    # return P.astype(np.int32)
    
    line1=gm.Line(gm.Point(A),gm.Point(B))
    line2=line1.perpendicular_segment(gm.Point(C))
     # Extract two points from the perpendicular line
    p1, p2 = line2.points  # sympy Points

    # Convert to NumPy integer arrays
    np_p1 = np.array([int(p1.x), int(p1.y)], dtype=np.int32)
    np_p2 = np.array([int(p2.x), int(p2.y)], dtype=np.int64)  # Different dtype example

    return (np_p1, np_p2)
def keep_largest_component(mask):
    """
    Filters out all but the largest connected component in a binary mask.
    """
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask  # No contours found, return original mask
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask and draw only the largest contour
    largest_mask = np.zeros_like(mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return largest_mask

# Example Usage

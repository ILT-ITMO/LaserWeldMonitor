import logging
import re
from pathlib import Path
from re import match

import cv2
import numpy as np


def bbox_center(coords: list[tuple[float, float]]):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (min(xs) + (max(xs) - min(xs)), min(ys) + (max(ys) - min(ys)))


def crop_four(img: np.ndarray):
    midy = img.shape[0] // 2
    midx = img.shape[1] // 2

    crops = []
    crops.append(img[0 : midy + 100, 0 : midx + 100, :])
    crops.append(img[0 : midy + 100, midx - 100 :, :])
    crops.append(img[midy - 100 :, 0 : midx + 100, :])
    crops.append(img[midy - 100 :, midx - 100 :, :])

    return crops
def get_red_mask(image):
    # Convert to HSV to better isolate red
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

def remove_text_from_line_area(image: np.ndarray, bbox: list[list[float]]):
 # Convert bbox points to integer numpy array
    pts = np.array(bbox, dtype=np.int32)

    # Make a copy to avoid modifying original image
    image_copy = image.copy()

    # Fill the polygon with black (0, 0, 0)
    cv2.fillPoly(image_copy, [pts], color=(0, 0, 0))

    return image_copy


def get_closest_horiz_line_width(image: np.ndarray, bbox: tuple[float, float, float, float]):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   # Red and Yellow in BGR space
    image=remove_text_from_line_area(image, bbox)
    lower_red = np.array([0, 0, 100])      # Lower bound for red
    upper_red = np.array([100, 100, 255])  # Upper bound for red
    
    lower_yellow = np.array([0, 100, 100])     # Lower bound for yellow
    upper_yellow = np.array([100, 255, 255])   # Upper bound for yellow
    
    # Create masks
    mask_red = cv2.inRange(image, lower_red, upper_red)
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    
    # Combine them
    mask = cv2.bitwise_or(mask_red, mask_yellow)

    img_blur = cv2.medianBlur(mask, 3, 0)
    # cv2.imshow("Grayscale Image", img_gray)

    # red_mask = get_red_mask(image)
    # print (red_mask)
    # kernel = np.ones((5, 5), np.uint8)

# Dilate the image
    # dilated = cv2.dilate(img_blur, kernel, iterations=2)
  
    ## first approach
    edges_mid = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)
    cv2.imwrite("gray_debug.jpg", mask)
    lines = cv2.HoughLinesP(
        edges_mid, 1, np.pi / 180, 30, minLineLength=100, maxLineGap=20
    )
    horiz_lines = []
    for i in range(lines.shape[0]):
        line = (lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
        if abs(line[1] - line[3]) == 0:
            horiz_lines.append(line)

    line_centers = np.array([(line[2] - line[0], line[1]) for line in horiz_lines])
    bbc = bbox_center(bbox)
    distances = np.linalg.norm(line_centers - bbc, axis=1)
    min_index = np.argmin(distances)
    selected_line = horiz_lines[min_index]
      # Convert endpoints to np.array format for render_line
    pt1 = np.array([selected_line[0], selected_line[1]])
    pt2 = np.array([selected_line[2], selected_line[3]])
    res=horiz_lines[min_index][2] - horiz_lines[min_index][0]
    ## second approach
    # horiz_lines = []
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     if abs(y1 - y2) <= 3:  # allow small tilt
    #         if x1 > x2:  # ensure order
    #             x1, x2 = x2, x1
    #         horiz_lines.append((x1, y1, x2, y2))

    # # Merge colinear and close lines
    # merged_lines = []
    # used = set()
    # for i in range(len(horiz_lines)):
    #     if i in used:
    #         continue
    #     x1, y1, x2, y2 = horiz_lines[i]
    #     line_group = [(x1, y1, x2, y2)]
    #     used.add(i)

    #     for j in range(i + 1, len(horiz_lines)):
    #         if j in used:
    #             continue
    #         x3, y3, x4, y4 = horiz_lines[j]
    #         if abs(y1 - y3) < 5:
    #             if not (x2 < x3 - 20 or x1 > x4 + 20):
    #                 line_group.append((x3, y3, x4, y4))
    #                 used.add(j)

    #     # Merge into one long line
    #     xs = [pt for line in line_group for pt in (line[0], line[2])]
    #     ys = [line[1] for line in line_group]
    #     merged_lines.append((min(xs), ys[0], max(xs), ys[0]))

    # # Pick the closest merged line
    # bbc = bbox_center(bbox)
    # line_centers = np.array([[(line[0] + line[2]) / 2, line[1]] for line in merged_lines])
    # distances = np.linalg.norm(line_centers - bbc, axis=1)
    # min_index = np.argmin(distances)

    # selected_line = merged_lines[min_index]
    # pt1 = np.array([selected_line[0], selected_line[1]])
    # pt2 = np.array([selected_line[2], selected_line[3]])
    # line_length = selected_line[2] - selected_line[0]

    # return line_length, (pt1, pt2)
    return res, (pt1, pt2)

def find_long_colored_line(image: np.ndarray, bbox: tuple[float, float, float, float] = None):
    # Optional: remove text over line if bbox is known
    if bbox is not None:
        x1, y1 = np.min(bbox, axis=0).astype(int)
        x2, y2 = np.max(bbox, axis=0).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Yellow range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create masks
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_red, mask_yellow)

    # Morphological closing to connect broken parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, image  # Nothing found

    # Find the longest contour by bounding box width
    max_line = max(contours, key=lambda c: cv2.boundingRect(c)[2])

    # Get bounding box and endpoints
    x, y, w, h = cv2.boundingRect(max_line)
    pt1 = np.array([x, y + h // 2])
    pt2 = np.array([x + w, y + h // 2])

    # Draw the line and label
    result = image.copy()
    cv2.arrowedLine(result, pt1, pt2, (0, 255, 0), thickness=4)
    cv2.arrowedLine(result, pt2, pt1, (0, 255, 0), thickness=4)
    cv2.putText(result, f"{w}px", (x + w // 2 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    return w, (pt1, pt2)
def match_units_text(res_ocr):
    """Match units text from EasyOCR results."""
    val_unit_text = None
    pred_index = None  # ← INITIALIZE WITH DEFAULT VALUE
    
    if res_ocr:
        for i, (bbox, text, confidence) in enumerate(res_ocr):
            text_clean = text.replace(" ", "")
            if re.match(r"\d+\s*(mm|nm|m|cm|μm)", text_clean):
                val_unit_text = text_clean
                pred_index = i
                break
    return val_unit_text, pred_index
def get_pixel_real_size(
    reader, image
) -> tuple[list[float, float], str]:
    """Finds real value, unit.

    Args:
        path (path): Image file path.

    Returns:
        tuple[float, str]: Real size of pixel side and its units.
    """

    # search on full image
    # res_ocr = reader.ocr(image)[0]
    res_ocr = reader.readtext(image)
    val_units_text, pred_id = match_units_text(res_ocr)

    if val_units_text:
        value, unit = re.findall("[a-z]+|[0-9]+", val_units_text)
        
        line_length_px, line = find_long_colored_line(image, res_ocr[pred_id][0])
        # line_length_px, line = get_closest_horiz_line_width(image, res_ocr[pred_id][0])
    else:  # search on crops
        crops = crop_four(image)
        for i, crop in enumerate(crops):
            res_ocr = reader.readtext(crop)
            # res_ocr = reader.ocr(crop)[0]
            val_units_text, pred_id = match_units_text(res_ocr)
            if val_units_text:
                value, unit = re.findall("[a-z]+|[0-9]+", val_units_text)
                line_length_px, line = find_long_colored_line(image, res_ocr[pred_id][0])
                
                # line_length_px, line = get_closest_horiz_line_width(crop, res_ocr[pred_id][0])

    unit = "μm" if unit == "m" else unit
    print ("v", value)
    print ("l", line_length_px)
    return int(value)/line_length_px, unit, line

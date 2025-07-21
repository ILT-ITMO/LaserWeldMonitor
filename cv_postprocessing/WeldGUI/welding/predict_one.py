import os
import torch
import cv2
import numpy as np
import re
import json
import math
from paddleocr import PaddleOCR
from ultralytics import YOLO
from src.contours import *
from src.ocr import get_pixel_real_size
from src.render import *
from src.gost import check_gosts
import pathlib
# Set environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_mask(model, img):
    H, W, _ = img.shape
    results = model(img, imgsz=[640], iou=0.4, conf=0.1, augment=True)
    mask = results[0].masks.data[0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (W, H))
    return mask

def analyze_single_image(config, image_path):
    # Initialize models
    ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
    model1 = YOLO(config['middle_part_path'])
    model2 = YOLO(config['plate_model_path'])

    # Read and prepare image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    key = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Process image
        im = cv2.imread(str(image_path))
        plot = im.copy()

        # Get pixel-to-real size conversion
        le, u, _ = get_pixel_real_size(ocr, im)

        # Get masks
        mask1 = get_mask(model1, im)
        mask1 = keep_largest_component(mask1)
        
        mask2 = get_mask(model2, im)
        mask2 = cv2.subtract(mask2, mask1)
        
        # Clean mask
        kernel = np.ones((16, 16))
        mask2 = cv2.erode(mask2, kernel, iterations=4)
        mask2 = cv2.dilate(mask2, kernel, iterations=4)
        mask2 = cv2.medianBlur(mask2, 5)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find contours
        main_object_con = n_max_contours(mask1)
        plate_part_cntrs = n_max_contours(mask2, n=2)
        
        # Process contours
        quad = [approximte_contour(q) for q in plate_part_cntrs]
        rect = [bounding_rectangle(q) for q in quad]
        case = 1 if len(rect) == 1 else 2
        
        # Find plate width lines
        main_sides_rect = plate_width_line(rect, main_object_con)
        (p11, p12), (p21, p22) = plate_width_line(quad, main_object_con)
        c1, c2 = line_intersection_contur([(p11, p21), (p12, p22)], main_object_con)
        
        # Draw on plot
        plot = cv2.polylines(plot, [c1], False, (255, 0, 255), 5)
        plot = cv2.polylines(plot, [c2], False, (255, 0, 255), 5)
        plot = cv2.line(plot, tuple(p11), tuple(p21), (255, 255, 0), 5)
        plot = cv2.line(plot, tuple(p12), tuple(p22), (255, 255, 0), 5)
        
        # Ensure consistent point ordering
        if p12[1] > p11[1]:
            p11, p12 = p12, p11
            p21, p22 = p22, p21
            c1, c2 = c2, c1

        # Calculate measurements
        res_d = []
        if case == 1:
            line = perpendicular_foot(p11, p21, p22)
            t = np.linalg.norm(line[0] - line[1])
        else:
            t = np.linalg.norm(main_sides_rect[0][0] - main_sides_rect[0][1])
        
        for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
            dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
            res_d.append(np.abs(dist))
        
        if case == 1:
            hs = res_d[1][1] * le
            hi = res_d[0][1] * le
            hg = 0
            he = res_d[1][0] * le
            hp = 0
            b_upper = (np.linalg.norm(p12 - p22) ** 2 - 
                      calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
            b_downer = (np.linalg.norm(p11 - p21) ** 2 - 
                       calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
            t = t - res_d[0][1]
        else:
            hi = 0
            hs = res_d[1][1] * le
            hg = res_d[0][0] * le
            he = res_d[1][0] * le
            hp = res_d[0][1] * le
            b_downer = (np.linalg.norm(p11 - p21) ** 2 - 
                       calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
            b_upper = (np.linalg.norm(p12 - p22) ** 2 - 
                      calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
        
        # Calculate misalignment
        misalignment = 0
        for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
            dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
            if case == 1:
                if p1 is not None:
                    for pr in zip(p1, p2):
                        length = np.linalg.norm(np.array(pr[1]) - np.array(pr[0])) * le
                        if math.isclose(length, hi, rel_tol=1e-1) or math.isclose(length, he, rel_tol=1e-1) or math.isclose(length, hs, rel_tol=1e-1):
                            plot = render_line(plot, pr, le, u)
                misalignment_top = compute_misalignment(p12, p22, line) * le
                misalignment_bottom = compute_misalignment(p11, p21, line) * le
                misalignment = max(misalignment_top, misalignment_bottom)
            else:
                if p1 is not None:
                    for pr in zip(p1, p2):
                        plot = render_line(plot, pr, le, u)
                misalignment_top = compute_misalignment(p12, p22, main_sides_rect[0]) * le
                misalignment_bottom = compute_misalignment(p11, p21, main_sides_rect[0]) * le
                misalignment = max(misalignment_top, misalignment_bottom)
        
        # Add misalignment text to plot
        plot = cv2.putText(plot, f'misalignment: {misalignment:.2f}{u}', 
                          (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        
        # Calculate area
        A = np.count_nonzero(mask1) * le * le
        
        # Create results dictionary
        result = {
            "b_upper": b_upper * le,
            "t": t * le,
            "A": A,
            "hg": hg,
            "he": he,
            "hp": hp,
            "hs": hs,
            "hm": misalignment,
            "hi": hi,
            "b_downer": b_downer * le,
        }
        
        # Check GOST standards
        gost_result = check_gosts(result)
        
        # Create masked image
        masked_image = render_mask(img, mask1)
        
        # Convert rendered images back to BGR for saving/display
        plot_bgr = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        masked_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        
        return result, gost_result, plot_bgr, masked_bgr

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == '__main__':
    # Load configuration
    config_path = pathlib.Path('.\welding\config.json').resolve()
    with open(config_path, 'r') as f:
        config = json.load(f)

    image_path = pathlib.Path(config['single_image_path']).resolve()
    output_path = pathlib.Path(config['single_output_path']).resolve()
    # Process the single image
    result, gost, rendered_img, masked_img = analyze_single_image(config, image_path)
    
    if result is not None:
        print("Measurement Results:")
        for k, v in result.items():
            print(f"{k}: {v:.2f}")
        
        print("\nGOST Compliance:")
        for k, v in gost.items():
            print(f"{k}: {v}")
        
        # Save rendered images
        cv2.imwrite(str(output_path / 'rendered_result.jpg'), rendered_img)
        cv2.imwrite(str(output_path / 'masked_result.jpg'), masked_img)
        print("\nRendered images saved as 'rendered_result.jpg' and 'masked_result.jpg'")
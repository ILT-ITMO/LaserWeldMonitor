import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import pathlib
import sys
import numpy as np
import json
import pandas as pd
import re
from paddleocr import PaddleOCR
from ultralytics import YOLO
import math
from welding.src.contours import *
from welding.src.ocr import get_pixel_real_size
from welding.src.render import *
from welding.src.gost import check_gosts
import argparse
# Set environment variable at the beginning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_mask(model, img):
    H, W, _ = img.shape
    results = model(img, imgsz=[640], iou=0.4, conf=0.1, augment=True)
    mask = results[0].masks.data[0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (W, H))
    return mask

def split_key(key):
    match = re.match(r"(\d+)([a-zA-Z]+)", key)
    if match:
        return (int(match.group(1)), match.group(2))
    return (float('inf'), '')

def process_single_image(key, img, image_path, model1, model2, ocr, output_masked, output_rendered, render, le=None, u=None, line=None):
    try:
        im = cv2.imread(os.path.join(str(image_path), key + '.jpg'))
        print(os.path.join(str(image_path), key + '.jpg'))
        plot = im.copy()
        # initialize UI output data 
        lines_data = {
            'contour_lines': [],       # c1, c2
            'plate_lines': [],         # p11-p21, p12-p22
            'main_sides': [],         # main_sides_rect
            'deviation_lines': [],     # deviation peaks
            'perpendicular_lines': [],  # any perpendicular lines
            'misalignment':[], #misalignment
            'mask_contours':[]
        }

        # le, u, line = get_pixel_real_size(ocr, im)

        mask1 = get_mask(model1, im)
        mask1 = keep_largest_component(mask1)
        mask_contours = []

        # Find contours to send them to UI 
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Simplify and store contours for UI 
        for cnt in contours:
            epsilon = 0.001 * cv2.arcLength(cnt, True)  # Adjust epsilon for simplification
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            mask_contours.append(approx.squeeze().tolist())  # Convert to list and remove single-dim entries

        # Add to lines_data
        lines_data['mask_contours'] = mask_contours
        # create mask of side parts of the plate
        mask2 = get_mask(model2, im)
        mask2 = cv2.subtract(mask2, mask1)
        
        # clear image from noise
        kernel = np.ones((16, 16))
        mask2 = cv2.erode(mask2, kernel, iterations=4)
        mask2 = cv2.dilate(mask2, kernel, iterations=4)
        mask2 = cv2.medianBlur(mask2, 5)  # Extra cleanup
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

      

        # find contours
        main_object_con = n_max_contours(mask1)  # all the contours for the main part
        plate_part_cntrs = n_max_contours(mask2, n=2)  # 2 contours for the plate
        
        # find bounding rectangles
        quad = [approximte_contour(q) for q in plate_part_cntrs]
        rect = [bounding_rectangle(q) for q in quad]
        
        if (len(rect) == 1):
            case = 1
        else:
            case = 2
            
        # approximate side lines
        main_sides_rect = plate_width_line(rect, main_object_con)
        (p11, p12), (p21, p22) = plate_width_line(quad, main_object_con)
        c1, c2 = line_intersection_contur([(p11, p21), (p12, p22)], main_object_con)
        plot = cv2.polylines(plot, [c1], False, (255, 0, 255), 5)
        plot = cv2.polylines(plot, [c2], False, (255, 0, 255), 5)
        plot = cv2.line(plot, tuple(p11), tuple(p21), (255, 255, 0), 5)  # bottom line
        plot = cv2.line(plot, tuple(p12), tuple(p22), (255, 255, 0), 5)  # top line
        
        if p12[1] > p11[1]:
            # Swap the points to ensure p12 is always above p11
            p11, p12 = p12, p11
            p21, p22 = p22, p21
            c1, c2 = c2, c1
        lines_data['contour_lines'].extend([c1.tolist(), c2.tolist()])
        
        # Store plate lines (p11-p21, p12-p22)
        lines_data['plate_lines'].extend([
            [p11.tolist(), p21.tolist()],
            [p12.tolist(), p22.tolist()]
        ])
        res_d = []
        if case == 1:
            line = perpendicular_foot(p11, p21, p22)
            line_right = perpendicular_foot(p11, p21, p12)
            t = np.linalg.norm(line[0] - line[1])
        else:
            t = np.linalg.norm(main_sides_rect[0][0] - main_sides_rect[0][1])
        
        # calculate distances between sides of second masks
        for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
            dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
            res_d.append(np.abs(dist))
           
        if (case == 1):
            plot = render_line(plot, line, le, u)
            lines_data['main_sides'].append([line[0].tolist(), line[1].tolist()])
            
            hs = res_d[1][1] * le
            hi = res_d[0][1] * le
            hg = 0
            he = res_d[1][0] * le
            hp = 0
            b_upper = (np.linalg.norm(p12 - p22) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
            b_downer = (np.linalg.norm(p11 - p21) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
            
            t = t - res_d[0][1]
        
        if (case == 2):
            plot = render_line(plot, main_sides_rect[0], le, u)

            lines_data['main_sides'].append([main_sides_rect[0][0].tolist(), main_sides_rect[0][1].tolist()])

            hi = 0
            hs = res_d[1][1] * le
            hg = res_d[0][0] * le
            he = res_d[1][0] * le
            hp = res_d[0][1] * le
            b_downer = (np.linalg.norm(p11 - p21) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
            b_upper = (np.linalg.norm(p12 - p22) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
        
        for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
            dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
            if (case == 1):
                if p1 is not None:
                    for pr in zip(p1, p2):
                        length = (np.linalg.norm(np.array(pr[1]) - np.array(pr[0]))) * le
                        if math.isclose(length, hi, rel_tol=1e-1) or math.isclose(length, he, rel_tol=1e-1) or math.isclose(length, hs, rel_tol=1e-1):
                            lines_data['deviation_lines'].append([pr[0].tolist(), pr[1].tolist()])
                            plot = render_line(plot, pr, le, u)
                misalignment_top = compute_misalignment(p12, p22, line) * le
                misalignment_bottom = compute_misalignment(p11, p21, line) * le
                misalignment = max(misalignment_top, misalignment_bottom)
            
            if (case == 2):
                if p1 is not None:
                    for pr in zip(p1, p2):
                        plot = render_line(plot, pr, le, u)
                        lines_data['deviation_lines'].append([pr[0].tolist(), pr[1].tolist()])
                misalignment_top = compute_misalignment(p12, p22, main_sides_rect[0]) * le
                misalignment_bottom = compute_misalignment(p11, p21, main_sides_rect[0]) * le
                misalignment = max(misalignment_top, misalignment_bottom)
        
        plot = cv2.putText(plot, f'misalignment: %.2f' % misalignment + u, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        lines_data['misalignment'].append(misalignment)
        A = np.count_nonzero(mask1) * le * le
        result = {
            "key": key,
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
        
        cv2.imwrite(str(output_rendered / f'{key}.jpg'), plot)
        cv2.imwrite(str(output_masked / f'{key}.jpg'), render_mask(img, mask1))
        print(f"Saved rendered image to: {output_rendered}")
        print(f"Saved masked image to: {output_masked}")
        
        # Verify files exist
        print(f"Rendered image exists: {os.path.exists(output_rendered)}")
        print(f"Masked image exists: {os.path.exists(output_masked)}")
        
        # Print image stats if they exist
        if os.path.exists(output_rendered):
            rendered_size = os.path.getsize(output_rendered)
            print(f"Rendered image size: {rendered_size} bytes")
            
        if os.path.exists(output_masked):
            masked_size = os.path.getsize(output_masked)
            print(f"Masked image size: {masked_size} bytes")

        return result, check_gosts(result), lines_data
    
    except cv2.error as e:
        print(f"error while proceeding {key}")
        print(e)
        return None, None

def process_all_images(config, ocr):

    results = {
    "images": [],
    "summary": None,  
         "csv_data": None 
        }

    # ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

    middle_part_path = pathlib.Path(config['middle_part_path']).resolve()
    plate_model_path = pathlib.Path(config['plate_model_path']).resolve()
    model1 = YOLO(middle_part_path)
    model2 = YOLO(plate_model_path)

    # prepare paths
    image_path = pathlib.Path(config['image_path']).resolve()
    output_path = pathlib.Path(config['output_path']).resolve()
    output_masked = output_path / "masked"
    output_rendered = output_path / "rendered"
    output_result = output_path / "props.csv"
    output_gosts = output_path / "gosts.csv"

    output_masked.mkdir(parents=True, exist_ok=True)
    output_rendered.mkdir(parents=True, exist_ok=True)

    if output_path.is_file():
        print("Output destination not a folder.")
        return None

    if not output_path.exists():
        output_path.mkdir()

    render = config['render']
    imgs = dict()

    # read images
    if image_path.is_dir():
        # print("id\n",image_path)
        
        for img_id in image_path.glob('**/*'):
            if img_id.suffix.lower() == '.json' or img_id.suffix.lower() == '.zip':
                continue
            # print("id\n",img_id)
            if img_id.is_file():
                # print("idd\n",img_id)
                        
                img = cv2.imread(str(img_id))
                # print("idd\n",img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs[img_id.stem] = img
    else:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[image_path.stem] = img
        image_path = image_path.parent

    res = []
    gosts = []

    scale_params = config.get('scale_params', {})
    default_le = scale_params.get('le', None)
    default_u = scale_params.get('unit', None)
    default_line = scale_params.get('reference_line', None)
    print("def",default_le)
    # ... rest of your existing code ...
    
    for key, img in imgs.items():
        # Use provided scale params or calculate new ones
        if default_le and default_u and default_line:
            le, u, line = default_le, default_u, np.array(default_line)
        else:
            le, u, line = get_pixel_real_size(ocr, img)
        print("le", le)
        result, gost_result, lines_data = process_single_image(
            key, img, image_path, model1, model2, ocr, 
            output_masked, output_rendered, render, le, u, line
        )
        if result is not None:
            res.append(result)
            gosts.append(gost_result)
            image_result = {
                "id": key,  # or any unique identifier
                "imageName": key,  # filename or identifier
                "result": result,
                # "gostResult": gost_result,
                "linesData": lines_data,
                "scaleParams": {
                    "le": le,
                    "u": u,
                    "line": line.tolist() if isinstance(line, np.ndarray) else line
                }
            }
            
            results["images"].append(image_result)
            

    # Save results to CSV files
    res_df = pd.DataFrame(res)
    gosts_df = pd.DataFrame(gosts)
    gosts_df.to_csv(output_gosts)
    
    # Apply the function to create a sorting key
    res_df['sort_key'] = res_df['key'].apply(split_key)
    
    # Sort the DataFrame based on the sorting key
    res_df = res_df.sort_values('sort_key')
    
    # Drop the temporary 'sort_key' column
    res_df = res_df.drop(columns=['sort_key'])
    
    # Save the sorted DataFrame back to a CSV
    res_df.to_csv(output_result, index=False)
    results["csv_data"] = {
        "properties": res_df.to_dict(orient='records')
        # "gosts": gosts_df.to_dict(orient='records')
    }
    return results
def main(config_path=None, config_dict=None, ocr=None):
    """
    Main entry point that can be called either with:
    - config_path (str/pathlib.Path): Path to config file (CLI usage)
    - config_dict (dict): Pre-loaded config dictionary (direct call)
    
    Returns:
        dict: Processing results (structure depends on your implementation)
    """
    # Handle input parameters
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        config_path = pathlib.Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(str(config_path), 'r') as json_data:
            config = json.loads(json_data.read())
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    # Process the images and return results
    results = process_all_images(config, ocr)
    
    # Ensure process_all_images returns a dict that can be JSON-serialized
    return results

if __name__ == '__main__':
    # CLI Interface (unchanged from original)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Call main with CLI arguments
    results = main(config_path=args.config)
    
    # Print results for CLI usage
    print(json.dumps(results, indent=2))
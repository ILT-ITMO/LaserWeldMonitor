import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2

import pathlib
import sys
sys.path.append(os.path.abspath("welding"))
print(os.getcwd())
import numpy as np
import json
import pandas as pd
import re

from paddleocr import PaddleOCR
from ultralytics import YOLO

import json     
import math
from src.contours import  *
from src.ocr import get_pixel_real_size
from src.render import *
from src.gost import check_gosts
import pandas as pd

def get_mask(model, img):
    H, W, _ = img.shape

    results = model(img,imgsz=[640],  iou=0.4,conf=0.1,augment=True )
    mask = results[0].masks.data[0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (W, H))

    return mask




# Define a function to split the 'key' into numeric and alphabetical parts
def split_key(key):
    match = re.match(r"(\d+)([a-zA-Z]+)", key)
    if match:
        return (int(match.group(1)), match.group(2))
    return (float('inf'), '')  # ssshandle case where the key does not match the expected pattern




def main(config):
    ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

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
        for img_id in image_path.glob('**/*'):
            if img_id.is_file():
                img =  cv2.imread(str(img_id))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs[img_id.stem] =  img
    else:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[image_path.stem] = img
        image_path = image_path.parent

    
    res = []
    gosts = []

    for key, img in imgs.items():
        try:
            im = cv2.imread(os.path.join(str(image_path), key+'.jpg'))
            print(os.path.join(str(image_path), key+'.jpg'))
            plot = im.copy()

         
            le, u, line = get_pixel_real_size(ocr, im)
 
            mask1 = get_mask(model1, im)
            mask1=keep_largest_component(mask1)
          
            # create mask of side parts of the plate
            mask2 = get_mask(model2, im)
            mask2 = cv2.subtract(mask2,mask1)
            
  # clear image from noise
            kernel = np.ones((16, 16))
            mask2 = cv2.erode(mask2, kernel, iterations=4)
            mask2 = cv2.dilate(mask2, kernel, iterations=4)
            mask2 = cv2.medianBlur(mask2, 5)  # Extra cleanup
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=3)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

            cv2.imwrite('mask2.jpg', mask2)


            # find contours
            main_object_con = n_max_contours(mask1) #all the contours for the main part, 
            plate_part_cntrs = n_max_contours(mask2, n=2) #2 contours for the plate 
            
    
            # find bounding rectangles
            quad = [approximte_contour(q) for q in plate_part_cntrs]
            rect = [bounding_rectangle(q) for q in quad]
            # print(len(rect))
            if (len(rect)==1):
                case=1
            else:
                case=2
            # print(case)
            #approximate side lines
            main_sides_rect = plate_width_line(rect, main_object_con)
            #Aveen Here: so here I see that we are always taking 4 points, but the correct solution is to take 3 if incomplete penetration 
            (p11, p12), (p21, p22) = plate_width_line(quad, main_object_con)
            c1, c2 = line_intersection_contur([(p11, p21), (p12, p22)], main_object_con)
            #here c1 and c2 are the 2 pink lines which are the closest parts between the main and the 
            # c1, c2 = approximte_contour(c1, epsilon=0.001), approximte_contour(c2, epsilon=0.001)
            plot = cv2.polylines(plot, [c1], False, (255, 0, 255), 5)
            plot = cv2.polylines(plot, [c2], False, (255, 0, 255), 5)
            plot = cv2.line(plot, tuple(p11), tuple(p21), (255, 255, 0), 5) # bottom line
            plot = cv2.line(plot, tuple(p12), tuple(p22), (255, 255, 0), 5) #top line 
            
            if p12[1] > p11[1]:
                # Swap the points to ensure p12 is always above p11
                p11, p12 = p12, p11
                p21, p22 = p22, p21
                c1, c2 = c2,c1
                # print(p11,"**", p12)
                
                        # print(p11)
            res_d = []
            if case==1:
                line = perpendicular_foot(p11, p21, p22)
                line_right=perpendicular_foot(p11, p21, p12)
                t = np.linalg.norm(line[0] - line[1])
            else:
                t = np.linalg.norm(main_sides_rect[0][0] - main_sides_rect[0][1])
            
            
            # calculate distances between sides of second masks
            for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
                
                # approximate upper and lower sides of middle part with straight lines
                # and then calculate maximum and minimum deviations between contours and projection on the line
                #Aveen Here: yeah here we also don't take into acount if it's 2 points down tehre or one
                # t = np.linalg.norm(main_sides_rect[0][0] - main_sides_rect[0][1])
                dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
                
                res_d.append(np.abs(dist))
        
            if (case==1):
                

                
             
                plot = render_line(plot,line , le, u)
             
                hs=res_d[1][1] * le
                hi=res_d[0][1] * le 
                hg=0
                he=res_d[1][0] * le
                hp=0
                b_upper = (np.linalg.norm(p12 - p22) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
                b_downer = (np.linalg.norm(p11 - p21) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
                # print(t*le)
                
                t=t-res_d[0][1]
       
            if (case==2):
          
                
                plot = render_line(plot, main_sides_rect[0], le, u)
                hi=0
                hs=res_d[1][1] * le
                hg= res_d[0][0] * le
                he=res_d[1][0] * le
                hp=res_d[0][1] * le
                b_downer = (np.linalg.norm(p11 - p21) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
                b_upper = (np.linalg.norm(p12 - p22) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5
            
            
            for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
                dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
                if (case==1):
            
                    if p1 is not None:
                            for pr in zip(p1, p2):
                                length = (np.linalg.norm(np.array(pr[1]) - np.array(pr[0])))*le
                                if math.isclose(length, hi, rel_tol=1e-1) or math.isclose(length, he, rel_tol=1e-1) or math.isclose(length, hs, rel_tol=1e-1):
                                    plot = render_line(plot, pr, le, u)
                    misalignment_top = compute_misalignment(p12, p22, line) * le
                    misalignment_bottom = compute_misalignment(p11, p21, line) * le
                    
                    misalignment = max(misalignment_top, misalignment_bottom)
                    # misalignment= compute_misalignment(p12, p22, line)* le
                                    
                    # misalignment = calculate_bias(line, line_right) * le
                                
                if (case==2):
                    if p1 is not None:
                        for pr in zip(p1, p2):
                            plot = render_line(plot, pr, le, u)
                    # remaining_sides_rect=get_remaining_sides(rect[0], rect[1])
                    # misalignment= compute_misalignment(p11, p21, main_sides_rect[0])* le
                    # misalignment = calculate_bias(*main_sides_rect) * le
                    misalignment_top = compute_misalignment(p12, p22, main_sides_rect[0]) * le
                    misalignment_bottom = compute_misalignment(p11, p21, main_sides_rect[0]) * le
                    
                    misalignment = max(misalignment_top, misalignment_bottom)
            
            plot = cv2.putText(plot, f'misalignment: %.2f' % misalignment + u, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

            
    
            print ("t\n", t)
            print ("le\n", le)
            print ("T\n", le*t)
            A= np.count_nonzero(mask1)*le*le
            res.append({
                "key": key,
                "b_upper": b_upper * le,
                "t": t * le,
                "A": A, 
                # "hu": hu,
                "hg": hg,
                "he": he,
                "hp": hp,
                "hs": hs,
                "hm": misalignment,
                "hi": hi,
                "b_downer": b_downer * le,
            })
            gosts.append(check_gosts(res[-1]))
            cv2.imwrite(str(output_rendered / f'{key}.jpg'), plot)
            cv2.imwrite(str(output_masked / f'{key}.jpg'), render_mask(img, mask1))
        except cv2.error as e:
            print(f"error while proceeding {key}")
            print(e)
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
if __name__ == '__main__':
    config_path = pathlib.Path('.\welding\config.json').resolve()
    # print(config_path)
    if not config_path.exists():
        print("Specified config file does not exists")

    with open(str(config_path), 'r') as json_data:
        config = json.loads(json_data.read())
        main(config)
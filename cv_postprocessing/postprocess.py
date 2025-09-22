import numpy as np
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
# from pixel_size.utils import get_pixel_real_size
from weld_processing.read_mask import plot_mask_and_point,return_points_and_size
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from tqdm import tqdm

def overlay_mask(image_path, mask, alpha=0.8):
    # Load the image
    # image = Image.open(image_path).convert("RGB")
    image = Image.open(image_path).convert("RGBA")
    # resized_image = image.resize(384,480)

    # Load the mask and convert it to an array
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = np.array(mask)

    # Create a colored version of the mask
    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)  # RGBA
    colored_mask[:, :, 0] = 255  # Red channel (color of the mask)
    colored_mask[:, :, 3] = mask * 255  # Alpha channel based on the mask

    # Convert the colored mask to an image
    colored_mask_image = Image.fromarray(colored_mask, 'RGBA')

    # Overlay the mask on the image with specified transparency
    overlay = Image.alpha_composite(image, colored_mask_image)

    # Save the result to the specified path
    overlay_path = "./predicted_masks/overlay/" + image_path.split('/')[3]
    overlay.save(overlay_path)
    
    # Convert overlay to format compatible with OpenCV (BGR) for further processing (if necessary)
    overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGR)
    cv2.imwrite(overlay_path, overlay_cv)

# Example usage:
# overlay_mask('./predicted_masks/images/00435-4194690285.png', './predicted_masks/masks/00435-4194690285.png')

def overlay_mask_jpg(image_path, masks, alpha=0.1):
    """
    Overlays a YOLO segmentation mask onto an image, ensuring only one connected polygon is drawn.
    
    Args:
        image_path (str): Path to the input image.
        masks (list): List of YOLO masks (polygon coordinates).
        alpha (float): Transparency level for the overlay.
    
    Saves:
        Processed image with overlay mask.
    """
    overlay_path = "./predicted_masks/overlay/" + image_path.split('/')[-1]  # Use last part of path
    
    # Open image in PIL
    img_pil = Image.open(image_path).convert("RGBA")
    overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    all_polygons = []

    # Collect all polygons from masks
    for mask in masks:
        polygon = mask.xy[0]  # Get polygon coordinates
        if len(polygon) >= 3:
            all_polygons.append(Polygon(polygon))  # Convert to Shapely polygon

    if not all_polygons:
        print("No valid polygons detected.")
        return  # Exit if no valid polygons

    # Merge polygons and keep only the largest one
    merged = MultiPolygon(all_polygons).buffer(0)  # Merge overlapping polygons
    if isinstance(merged, MultiPolygon):
        largest_polygon = max(merged.geoms, key=lambda p: p.area)  # Keep the largest polygon
    else:
        largest_polygon = merged

    # Convert the largest polygon back to a format usable by ImageDraw
    polygon_coords = list(largest_polygon.exterior.coords)

    # Draw the largest polygon
    overlay_draw.polygon(polygon_coords, fill=(0, 255, 0, 50))

    # Draw a centroid marker
    centroid = largest_polygon.centroid
    circle_radius = 5
    left_up_point = (centroid.x - circle_radius, centroid.y - circle_radius)
    right_down_point = (centroid.x + circle_radius, centroid.y + circle_radius)
    overlay_draw.ellipse([left_up_point, right_down_point], fill=(255, 0, 0))

    # Merge overlay with original image
    img_pil = Image.alpha_composite(img_pil, overlay)

    # Convert PIL image to OpenCV format and save
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, frame)

    print(f"Overlay saved: {overlay_path}")

def process_yolo_masks(masks):
    """
    Takes YOLO segmentation masks and returns a single connected binary mask.
    
    Args:
        masks (tensor): YOLOv8 segmentation mask tensor.
        
    Returns:
        final_mask (numpy.ndarray): Processed binary mask with only one connected component.
    """
    if masks is None or len(masks) == 0:
        return None  # No mask detected

    # Convert YOLO mask tensor to NumPy
    masks = masks.data.cpu().numpy()  # Shape: (num_masks, H, W)

    # Merge all masks into a single binary mask
    merged_mask = np.max(masks, axis=0)  # Combine multiple masks into one

    # Convert to uint8 format for OpenCV
    merged_mask = (merged_mask * 255).astype(np.uint8)

    # Apply morphological closing to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest connected component
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        final_mask = np.zeros_like(merged_mask)
        cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = merged_mask  # If no contours found, keep original

    return final_mask


def save_prediction_masks(train):
    PATH_Images="./datasets/dataset/valid/images"
    PATH_Masks="./predicted_masks/masks"
    
    list_img=[img for img in os.listdir(PATH_Images) if img.endswith('.jpg')==True]
  
    model_path = Path(train)  # Adjust path as needed
    model = YOLO(model_path)
    for i in tqdm(list_img):
        img_path=PATH_Images+"/"+ i
        # Load the model (replace 'best.pt' with your actual model file name)
        print(img_path)
        # Load an image
        img = cv2.imread(img_path)

            # Resize the image
        # print(img)
        H, W,_ = img.shape
        # conf_threshold=0.0001, iou_threshold=0.5
        results = model(img,  iou=0.5,conf=0.0001, verbose=False )
        # high_res = model.predict(img, imgsz=1280, conf=0.2)
        # low_res = model.predict(img, imgsz=320, conf=0.15)
        # results= high_res+low_res;
        # mask= results[0].masks.data[0].cpu().numpy() * 255
        mask= process_yolo_masks(results[0].masks)
        # print(mask.shape[0]/mask.shape[1])
        # mask = cv2.resize(mask, (W, H))
        # print(H/W)
    
        mask_path=PATH_Masks+'/'+ i
 
        cv2.imwrite(mask_path, mask)
        # print(mask_path, img_path)
        overlay_mask_jpg(img_path,results[0].masks[0])

      
import shutil
def add_to_dataset():
    path_images = './tmp/images/'
    path_masks= './tmp/masks/'
    path_masks_predicted='./predicted_masks/masks/'
    path_images_predicted='./predicted_masks/images/'
    path_overlay='./predicted_masks/overlay/'
    list_img=[img for img in os.listdir(path_overlay) if img.endswith('.jpg')==True]
    for i in tqdm(list_img):
        img_path_predicted=path_images_predicted+"/"+ i
        mask_path_predicted=path_masks_predicted+"/"+ i
          # Move predicted image to the image directory
        if os.path.exists(img_path_predicted):
            shutil.move(img_path_predicted, path_images)
        
        # Move predicted mask to the masks directory
        if os.path.exists(mask_path_predicted):
            shutil.move(mask_path_predicted, path_masks)
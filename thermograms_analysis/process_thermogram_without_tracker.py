from modules import detect_spatters, detect_welding_zone, filter_spatters
import numpy as np
import cv2
from tqdm import tqdm
import json


R = 30  # radius to filter arround the welding zone


def process_thermogram(path: str) -> int:
    frames = np.load(path)
    temp_frames = frames.copy()
    # t_min = 1000
    # t_max = 2000
    # frames = np.clip(frames, t_min, t_max)
    # frames -= t_min
    # frames = frames / (t_max - t_min)
    t_min = frames.min()
    t_max = frames.max()
    frames -= t_min
    frames = frames / (t_max - t_min)
    frames *= 255
    frames = frames.astype(np.uint8)
    k = 0

    #print(path)
    features = {'size': [], 'temp': [], 'n_spatters': [], 'welding_zone_temp': [], }

    for frame, t_frame in tqdm(zip(frames, temp_frames), total=len(frames)):
        #print(k)
        k += 1
        pts = np.zeros((frame.shape[0], frame.shape[1]) + (3,))  # for drawing

        try:
            center, ellips = detect_welding_zone(frame)
            frame = cv2.ellipse(frame, ellips, (255,), 1)
            x, y, w, h = center
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            features['welding_zone_temp'].append(t_frame[y1:y2, x1:x2].mean())
        except:
            # print('No welding zone detected')
            features['welding_zone_temp'].append(0)

        boxes = detect_spatters(frame)
        try:
            boxes = filter_spatters(boxes, center, R)
            #boxes = remove_reflection(boxes, center, DX_REFL, DY_REFL, R_REFL)
            #boxes = remove_reflection(boxes, center, DX_TR, DY_TR, R_TR)
            x, y, w, h = center
            pts = cv2.circle(pts, (int(x), int(y)), R, (130, 130, 130), 1)
        except:
            pass
            # print('Cannot filter boxes')
            

        size = []
        temp = []
        for pt in boxes:
            x, y, w, h = pt
            size.append(w * h)
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            temp.append(t_frame[y1:y2, x1:x2].mean())
            pts = cv2.circle(pts, (int(x), int(y)), 4, (0, 0, 255), 1)  

        if len(boxes):
            features['size'].append(np.array(size).mean())
            features['temp'].append(np.array(temp).mean())
            features['n_spatters'].append(boxes.shape[0])
        else:
            features['size'].append(0)
            features['temp'].append(0)
            features['n_spatters'].append(0)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        frame = np.concatenate((frame, pts), axis=1).astype(np.uint8)
        cv2.imshow('thermogram', frame)
        if cv2.waitKey(0) == ord('q'):
           break
    cv2.destroyAllWindows()
    return features


# with open('thermograms_analysis/metrics/metrics_40.json', 'r') as f:
#     meta = json.load(f)

# out = {}
# for t in tqdm(meta.keys(), total=len(meta.keys())):
#     features = process_thermogram(f'thermograms_analysis/data/{t}')
#     out[t] = features


# with open('thermograms_analysis/metrics/without_tracker.json', 'w') as f:
#     json.dump(out, f)


process_thermogram('thermograms_analysis/data/thermogram_27.npy')
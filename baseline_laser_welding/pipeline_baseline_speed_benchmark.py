import time
from models import Conv2DClassifier, resnet3d, HybridCNNTransformerMultilabelV2
import numpy as np
import cv2
import torch
import scipy.signal as sig
from tqdm import tqdm

HIDDEN_LAYERS = 256
WINDOW_SIZE = 64
N_RUNS = 100


model = HybridCNNTransformerMultilabelV2(4, 64, 4)
model.eval()

frames = frames = np.load('dataset/thermogram_17.npy')
temp_frames = frames.copy()
t_min = frames.min()
t_max = frames.max()
frames -= t_min
frames = frames / (t_max - t_min)
frames *= 255
frames = frames.astype(np.uint8)
print(frames.shape)


R = 40  # radius to filter arround the welding zone
DX_REFL, DY_REFL, R_REFL = -80, -10, 20  # parameters of reflection to remove
DX_TR, DY_TR, R_TR = 40, -40, 15  # parameters of trace to remove


def min_loc_LoG(img, k_size = 9, sigma = 1.8):
    """
    Perform min-loc-LoG filtering of grayscale image img
    Sungho K. Min-local-LoG Filter for Detecting Small Targets in 
    Cluttered Background // Electronics Letters. 
    – 2011. – Vol. 47. – № 2. – P. 105-106. DOI: 10.1049/el.2010.2066.

    sigma - std of gaussian
    k_size - size of kernel
    """
    x = np.arange(k_size).reshape(1, k_size)
    y = np.arange(k_size).reshape(k_size, 1)
    # generate fE (positive X)
    fE = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fE[fE > 0] = fE[fE > 0] / fE[fE > 0].sum()
    fE[fE < 0] = fE[fE < 0] / (-fE[fE < 0].sum())
    # generate fS (positive Y)
    fS = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fS[fS > 0] = fS[fS > 0] / fS[fS > 0].sum()
    fS[fS < 0] = fS[fS < 0] / (-fS[fS < 0].sum())
    # generate fW
    x = - np.fliplr(x)
    fW = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fW[fW > 0] = fW[fW > 0] / fW[fW > 0].sum()
    fW[fW < 0] = fW[fW < 0] / (-fW[fW < 0].sum())
    # generate fN
    y = - np.flipud(y)
    fN = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fN[fN > 0] = fN[fN > 0] / fN[fN > 0].sum()
    fN[fN < 0] = fN[fN < 0] / (-fN[fN < 0].sum())
    # perform 2D convolution with kernels
    def move(img, x, y):
        move_matrix = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv2.warpAffine(img, move_matrix, dimensions)

    Ie = sig.convolve2d(move(img, 4, 0), fE, mode = "same")
    Is = sig.convolve2d(move(img, 0, 4), fS, mode = "same")
    Iw = sig.convolve2d(move(img, -4, 0), fW, mode = "same")
    In = sig.convolve2d(move(img, 0, -4), fN, mode = "same")
    f = np.dstack((Ie, Is, Iw, In))
    fmap = np.min(f, axis = 2)
    #return (fmap / fmap.max() * 255).astype(np.uint8)
    return fmap

        
def detect_spatters(frame):
    filtered = min_loc_LoG(frame, 9)
    filtered = ((filtered > 2) * 255).astype(np.uint8)
    c, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3]]) for i,_ in enumerate(c)]
    contours = np.array(contours)
    if not len(contours):
        return ()
    wh = contours[:, 2:4] - contours[:, :2]
    contours[:, :2] = contours[:, :2] + wh / 2
    contours[:, 2:4] = wh
    contours = contours[np.all(wh < 5, axis=1)]
    return contours


def detect_welding_zone(frame):
    filtered = ((frame > 20) * 255).astype(np.uint8)
    c, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3]]) for i,_ in enumerate(c)]
    center = np.array(center)
    wh = center[:, 2:4] - center[:, :2]
    center[:, :2] = center[:, :2] + wh / 2
    center[:, 2:4] = wh
    areas = wh[:, 0] * wh[:, 1]
    max_id = areas.argmax()
    center = center[max_id]
    if areas[max_id] > 10:
        return center, cv2.fitEllipse(c[max_id])
    else:
        raise ValueError("Can't detect welding zone")


def filter_spatters(boxes, center, R):
    x, y = center[:2]
    contours = boxes[(boxes[:, 0] - x) ** 2 + (boxes[:, 1] - y) ** 2 > R ** 2]
    return contours



def process_frame_pipeline(frame, t_frame):
    # get features
    features = {}
    center, ellips = detect_welding_zone(frame)
    x, y, w, h = center
    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    features['welding_zone_temp'] = np.array([t_frame[y1:y2, x1:x2].mean(), ] * WINDOW_SIZE)
    boxes = detect_spatters(frame)
    size = []
    temp = []
    for pt in boxes:
        x, y, w, h = pt
        size.append(w * h)
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        temp.append(t_frame[y1:y2, x1:x2].mean())
    features['size'] = np.array([np.array(size).mean(),] * WINDOW_SIZE)
    features['temp'] = np.array([np.array(temp).mean(),] * WINDOW_SIZE)
    features['n_spatters'] = np.array([boxes.shape[0],] * WINDOW_SIZE)
    # sliding window
    # feat_vector = [
    #     features['welding_zone_temp'].mean(),
    #     features['welding_zone_temp'].std(),
    #     # np.median(features['welding_zone_temp']),
    #     # np.sqrt(np.mean(features['welding_zone_temp']**2)),
    #     features['size'].mean(),
    #     features['size'].std(),
    #     # np.median(features['size']),
    #     # np.sqrt(np.mean(features['size']**2)),
    #     features['temp'].mean(),
    #     features['temp'].std(),
    #     # np.median(features['temp']),
    #     # np.sqrt(np.mean(features['temp']**2)),
    #     features['n_spatters'].mean(),
    #     features['n_spatters'].std(),
    #     # np.median(features['n_spatters']),
    #     # np.sqrt(np.mean(features['n_spatters']**2)),
    # ]
    feat_vector = [
        features['welding_zone_temp'],
        features['size'],
        features['temp'],
        features['n_spatters']
    ]
    feat_vector = torch.tensor(feat_vector).t().unsqueeze(0).float()
    # classify
    with torch.no_grad():
        scores = model(feat_vector).squeeze().sigmoid()



start = time.time()
for _ in tqdm(range(N_RUNS), total=N_RUNS):
    process_frame_pipeline(frames[1000], temp_frames[1000])

elapsed = time.time() - start
print(f"Processing speed of pipeline: {elapsed / N_RUNS * 1000:.3f} ms")


cnn = Conv2DClassifier(1, 4)

def process_frame_baseline(frame):
    inp = torch.from_numpy(np.array([frame, ] * 1)).unsqueeze(0) / 255
    with torch.no_grad():
        scores = cnn(inp.float()).squeeze().sigmoid()

# cnn = resnet3d('resnet18')

# def process_frame_baseline(frame):
#     frames = torch.from_numpy(np.array([frame, ] * 32) / 255)
#     inp = torch.permute(frames[..., None].float(), (3, 0, 1, 2)).unsqueeze(0)
                           
#     with torch.no_grad():
#         scores = cnn(inp.float()).squeeze().sigmoid()

start = time.time()
for _ in tqdm(range(N_RUNS), total=N_RUNS):
    process_frame_baseline(frames[1000])

elapsed = time.time() - start
print(f"Processing speed of baseline: {elapsed / N_RUNS * 1000:.3f} ms")
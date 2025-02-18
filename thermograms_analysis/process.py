import numpy as np
import cv2
import scipy.signal as sig
from typing import List, Dict, Tuple, Union
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from tqdm import tqdm


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


def remove_reflection(boxes, center, dx, dy, r):
    x, y = center[:2]
    x += dx
    y += dy
    contours = boxes[(boxes[:, 0] - x) ** 2 + (boxes[:, 1] - y) ** 2 > r ** 2]
    return contours


class NearestTracker:
    """
Class Description:
This class is used to track points in a two-dimensional space. It assigns unique IDs to new points and keeps track of the position of these points over time. It also has the ability to update the position of these points based on new detections and filter out points based on a distance threshold.

Methods:
- __init__: This method is a constructor and it's used to initialize the instance of the class. It sets up the 'tracked' attribute as an empty dictionary, 'max_id' as zero and 'max_d' as 30. 

    Args:
        self (object): A reference to the instance of the class.

    Returns:
        None

- step: The 'step' function updates the tracked points based on the new detections. The function first checks if there are points that are currently being tracked. If not, the new detections are assigned to the tracked points. In case there are tracked points, the function tries to match the newly detected points with the previously tracked ones. Unmatched new points are assigned new IDs, and unmatched old points are discarded. The function also filters the matched points based on a distance threshold and updates the tracked points accordingly.
        
    Args:
        pts (array-like): Two dimensional array-like object representing the new detected points. Each row should represent a point in 2D space.

    Returns:
        dict: Returns a dictionary with point IDs as keys and the corresponding points as values. The dictionary is a deep copy of the updated tracked points. """
    def __init__(self):
        """
    Initializes the instance of the class.

    This method is a constructor and it's used to initialize the instance of the class. 
    It sets up the 'tracked' attribute as an empty dictionary, 'max_id' as zero and 'max_d' as 30.

    Args:
        self (object): A reference to the instance of the class.
        
    Returns:
        None
    """
        self.tracked = {}
        self.max_id = 0
        self.max_d = 30
    
    def step(self, pts):
        """
        The 'step' function updates the tracked points based on the new detections. 
        
        The function first checks if there are points that are currently being tracked. If not, the new detections are assigned to the tracked points. In case there are tracked points, the function tries to match the newly detected points with the previously tracked ones. Unmatched new points are assigned new IDs, and unmatched old points are discarded. The function also filters the matched points based on a distance threshold and updates the tracked points accordingly.
        
        Args:
            pts (array-like): Two dimensional array-like object representing the new detected points. Each row should represent a point in 2D space.

        Returns:
            dict: Returns a dictionary with point IDs as keys and the corresponding points as values. The dictionary is a deep copy of the updated tracked points. 
        """
        if not len(self.tracked):
            self.tracked = {i: pt for i, pt in enumerate(pts)}
            self.max_id = len(self.tracked)
            return deepcopy(self.tracked)
        
        ids_not_matched = []  # list of ids not matched from previous step
        pts_not_matched = []  # list of new kpts to assign new ids

        # Match tracked pts with new detections
        new_pts = np.expand_dims(pts[:, :2], 1)
        pts_ = np.array(list(self.tracked.values()))
        old_pts = np.expand_dims(pts_[:, :2], 0)
        diff = np.sqrt(np.square(new_pts - old_pts).sum(axis=-1))
        a = (diff < self.max_d).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            x, y = linear_sum_assignment(diff)
            matched_indices = np.array(list(zip(x, y)))
        
        # Select unmatched detections and tracks
        for d, det in enumerate(pts):  # iterate over detected kpts
            if d not in matched_indices[:, 0]:
                pts_not_matched.append(det)  # collect unmatched new points to assign new ids

        for t, (key, value) in enumerate(self.tracked.items()):  # iterate over tracked kpts
            if t not in matched_indices[:, 1]:
                ids_not_matched.append(key)  # collect unmatched old points to discard

        # Filter matched detections by distance
        matched = []
        keys = list(self.tracked.keys())
        for x, y in matched_indices:
            if diff[x, y] > self.max_d:
                pts_not_matched.append(pts[x])
                ids_not_matched.append(keys[y])
            else:
                matched.append((x, y))

        # Replace matched detections
        if len(matched) != 0:
            for x, y in matched:
                self.tracked[list(self.tracked.keys())[y]] = pts[x]

        # Remove unmatched tracks
        for id in ids_not_matched:
            self.tracked.pop(id)
        # Add unmatched detections
        for det in pts_not_matched:
            self.tracked[self.max_id] = det
            self.max_id += 1
        
        return deepcopy(self.tracked)


class Tracker:
    """
This class is used for tracking keypoints in a sequence of frames. It maintains a history of keypoints detected 
in the last three frames, and uses methods for calculating distances and angles to match new detections with 
existing tracks or create new tracks.

Methods:
    __init__(self): Initializes several attributes including maximum distance between joint points ('max_d'), 
    maximum delta angle to match points ('delta_angle'), a dictionary to store tracked points with their last 
    2 states ('tracked'), a list to store keypoints from the last 3 frames ('kpts'), and the maximum assigned 
    ID ('max_id').

    __compute_angle(pt1, pt2, ver): Computes the angle between two points (pt1 and pt2) with respect to a vertex 
    (ver) in a 2D space.

    __compute_distance(pt1, pts2): Calculates the Euclidean distance between a point 'pt1' and each point in 'pts2'.

    __match_pts(pts, pt1, pt2, used_ids): Iterates over a list of detected points, computes the distances between 
    these points and a given pair of points, and then finds the optimal and minimum distances.

    step(pts): Accumulates keypoints from the last three frames, matches new detections with existing tracks, 
    combines detections into new tracks and updates existing tracks. If no keypoints are provided or there are 
    less than three frames, returns an empty tuple."""
    def __init__(self):
        """
    Initializer method for this class.

    This method initializes several attributes for the class. The 'max_d' attribute is used to set the maximum 
    distance between joint points. The 'delta_angle' attribute sets the maximum delta angle to match points, 
    in degrees. The 'tracked' attribute is a dictionary that stores tracked points with their last 2 states. 
    The 'kpts' attribute is a list that stores keypoints from the last 3 frames. The 'max_id' attribute is used 
    to store the maximum assigned ID.

    Args:
        self (object): A reference to the instance of the class.

    Returns:
        None
    """
        self.max_d = 30  # maximum distance between joint points
        self.delta_angle = 20 # maximum delta angle 180 +- to match pts (in degrees)
        self.tracked = {}  # dict with tracked points (last 2 states stored)
        self.kpts = []  # kpts from last 3 frames
        self.max_id = 0  # max assigned ID

    @staticmethod
    def __compute_angle(pt1: np.ndarray, pt2: np.ndarray, ver: np.ndarray) -> float:
        """
    Computes the angle between two points with respect to a vertex.

    This method computes the angle between two points (pt1 and pt2) with respect to a vertex (ver) 
    in a 2D space. It uses the dot product to find the cosine of the angle, and then converts it to degrees.

    Args:
        pt1 (np.ndarray): The first point in 2D space.
        pt2 (np.ndarray): The second point in 2D space.
        ver (np.ndarray): The vertex point in 2D space.

    Returns:
        float: The angle in degrees between pt1 and pt2 with respect to ver.
    """
        cosine = np.dot(pt1[:2] - ver[:2], pt2[:2] - ver[:2]) / (np.linalg.norm(pt1[:2] - ver[:2]) * np.linalg.norm(pt2[:2] - ver[:2]))
        angle = np.degrees(np.arccos(cosine - np.sign(cosine) * 10e-6))
        return angle

    @staticmethod
    def __compute_distance(pt1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(pts2 - pt1).sum(axis=-1))
    
    def __match_pts(self, pts: Union[np.ndarray, List[np.ndarray]], pt1: np.ndarray, pt2: np.ndarray,
                    used_ids: List[int]) -> Tuple[Union[int, None], float, Union[int, None], float]:
        """
    Compute the optimal and minimum distance between a set of points and a given pair of points.

    This method iterates over a list of detected points, computes the distances between these points 
    and a given pair of points, and then finds the optimal and minimum distances. The optimal distance 
    is the smallest distance that meets specific conditions related to the relative distances and the angle 
    formed by the points. The minimum distance is simply the smallest computed distance. Points that 
    have been previously used are skipped in the iteration.

    Args:
        pts (Union[np.ndarray, List[np.ndarray]]): The detected points to calculate distances from.
        pt1 (np.ndarray): The first point of the pair to calculate distances to.
        pt2 (np.ndarray): The second point of the pair to calculate distances to.
        used_ids (List[int]): A list of indices of points in 'pts' that have been previously used.

    Returns:
        Tuple[Union[int, None], float, Union[int, None], float]: A tuple containing four elements:
            - The index in 'pts' of the point with the optimal distance, or None if no such point is found.
            - The optimal distance, or 181.0 if no such distance is found.
            - The index in 'pts' of the point with the minimum distance, or None if no such point is found.
            - The minimum distance, or 1000 if no such distance is found.
    """
        d1 = self.__compute_distance(pt1, pt2)  # compute distance between them
        # Init optimal
        optimal_id = None
        best_diff = 181.0

        # Init minimum
        min_id = None
        min_d = 1000

        # Iterate over detected kpts
        for det_id, det in enumerate(pts):
            # Skip if this detection was used
            if det_id in used_ids:
                continue
            d2 = self.__compute_distance(det, pt2)  # compute distance between last pt in track and detection
            # Find minimum distance
            if d2 < min_d:
                min_d = d2
                min_id = det_id
            # Filter detection by specific conditions (distance)
            if d2 > self.max_d:  # distance is less than max
                continue
            if min(d1, d2) / max(d1, d2) < 0.5:
                continue  # distance is nearly same as distance between previous kpts (optional)
            
            # Filter detection by angle
            angle = self.__compute_angle(pt1, det, pt2)  # compute angle between three pts
            diff = 180 - angle
            if diff > self.delta_angle:
                continue  # three pts don't form a straight line

            # If all filters passed, identify the most optimal match by criterion
            # TODO: try to change criterion
            if diff < best_diff:
                best_diff = diff
                optimal_id = det_id
        return optimal_id, best_diff, min_id, min_d

    def step(self, pts: Union[np.ndarray, None]) -> Union[Tuple, Dict[int, List[List[np.ndarray]]]]:
        # Accumulate kpts from 3 last frames
        if not len(pts):
            try:
                self.kpts.pop(0)
            except:
                pass
            return ()
        self.kpts.append(pts)
        if len(self.kpts) < 3:
            return ()
        
        new_tracks = []  # list for matched tracks
        used_ids_prev = []  # used pts from previous frame
        used_ids = []  # used pts from current frame
        unmatched_ids = []  # unmatched tracks

        # Match new detections with existed tracks
        if len(self.tracked):
            for id, track in self.tracked.items():  # iterate over existing tracks
                pt1 = track[-2][:2]  # get 2 last points from track
                pt2 = track[-1][:2]
                # Match
                optimal_id, _, min_id, min_d = self.__match_pts(pts[:, :2], pt1, pt2, used_ids)
                # Update track if match found
                if optimal_id is not None:
                    self.tracked[id].pop(0)
                    self.tracked[id].append(pts[optimal_id])
                    used_ids.append(optimal_id)
                elif min_id is not None and min_d < 2:
                    self.tracked[id].pop(0)
                    self.tracked[id].append(pts[min_id])
                    used_ids.append(min_id)
                else:
                    unmatched_ids.append(id)
            
            for id in unmatched_ids:
                self.tracked.pop(id)

        # Combine detections into new tracks
        pts1 = self.kpts[0]  # kpts from prev prev frame
        pts2 = self.kpts[1]  # kpts from prev frame
        if len(pts1) and len(pts2):
            # Iterate over prev prev pts
            for id1, pt1 in enumerate(pts1):
                optimal_id_2 = None
                optimal_id_3 = None
                best_diff_2 = 181.0
                # Iterate over prev kpts
                for id2, pt2 in enumerate(pts2):
                    if id2 in used_ids_prev:
                        continue  # skip if already matched
                    if self.__compute_distance(pt1[:2], pt2[:2]) > self.max_d:
                        continue  # check distance condition
                    optimal_id, best_diff, _, _ = self.__match_pts(pts[:, :2], pt1[:2], pt2[:2], used_ids)  # find best match
                    if optimal_id is None:
                        continue  # next if match not found
                    # Find optimal combination
                    if best_diff < best_diff_2:
                        optimal_id_3 = optimal_id
                        optimal_id_2 = id2
                        best_diff_2 = best_diff
                
                if (optimal_id_3 is not None) and (optimal_id_2 is not None):
                    # Update new combined tracks and used ids
                    new_tracks.append([pt1, pts2[optimal_id_2], pts[optimal_id_3]])
                    used_ids_prev.append(optimal_id_2)
                    used_ids.append(optimal_id_3)

        # Update tracks
        for track in new_tracks:
            self.tracked[self.max_id] = track
            self.max_id += 1

        # Update detections from previous iterations
        prev_kpts = []
        cur_kpts = []
        for id, pt in enumerate(pts):
            if id not in (used_ids):
                cur_kpts.append(pt)
        for id, pt in enumerate(pts2):
            if id not in (used_ids_prev):
                prev_kpts.append(pt)
        self.kpts = [prev_kpts, cur_kpts]
        return self.tracked.copy()


class FeatureExtractor:
    """
    This class is responsible for storing and computing metrics related to spatters such as welding zone temperature, 
    total number of spatters, mean size, temperature, cooling speed and velocity of spatters per frame, and new spatter
    appearance rate.

    The class is initialized with a specific window size and an empty dictionary to store metrics. The window size and 
    metrics dictionary are used to compute overall metrics based on the set metrics and the specified interpolation method.

    Methods
    --------
    __init__(self, w_size: int = 10):
        Initializes the object with a specific window size and an empty dictionary to store metrics.

    compute_results(self, use_interpolation: bool = False) -> Dict[str, float]:
        Computes overall metrics based on the set metrics and the specified interpolation method. 

    append(self, tracks: Dict[int, List[List[np.ndarray]]], t_frame: np.ndarray, center: np.ndarray):
        Processes the tracks and frame data to calculate various metrics related to spatters such as welding zone temperature, 
        total number of spatters, mean size, temperature, cooling speed and velocity of spatters per frame, and new spatter
        appearance rate."""
    def __init__(self, w_size: int = 10):
        """
    Initialize the object with a specific window size and an empty dictionary to store metrics.

    Args:
        self: The instance of the class.
        w_size (int, optional): The size of the window for which metrics are to be stored. Default is 10.

    Returns:
        None
    """
        self.metrics = {'total_spatters': [], 'velocity': [], 'size': [], 'temp': [], 'cooling_speed': [], 
                        'appearance_rate': [], 'n_spatters': [], 'welding_zone_temp': []}  # dict to store metrics
        self.last_tracks = None
        self.last_t_frame = None
        self.frames_to_interpolate = w_size

    def compute_results(self, use_interpolation: bool=False) -> Dict[str, float]:
        """
        Computes overall metrics based on the set metrics and the specified interpolation method.

        This method breaks down the metrics into four zones (A, B, C, D) and calculates the maximum, 
        minimum, and mean values for each metric in each zone. If interpolation is not used, 
        calculations are performed on the entire data set. If interpolation is used, calculations are 
        performed on smaller chunks of the data set, and the results are stored in a list for each metric 
        in each zone.

        Args:
            use_interpolation (bool): Whether or not to use interpolation when calculating metrics. 
            Defaults to False.

        Returns:
            Dict[str, float]: A dictionary where the keys are strings representing the metric and zone 
            (e.g., "total_spatters_A", "mean_appearance_rate_B"), and the values are either floats 
            representing the calculated metric value, or lists of floats if interpolation was used.
        """
        total = len(self.metrics['total_spatters'])
        chunk_size = total // 4
        overall_metrics = {}
        if not use_interpolation:
            for key in self.metrics.keys():
                for i, zone in enumerate(('A', 'B', 'C', 'D')):
                    data = np.array(self.metrics[key][i * chunk_size : (i + 1) * chunk_size])
                    if key == 'total_spatters':
                        overall_metrics[f"{key}_{zone}"] = int(data.max() - data.min())
                    elif key in ('appearance_rate', 'n_spatters'):
                        overall_metrics[f"mean_{key}_{zone}"] = float(data.mean())
                        overall_metrics[f"max_{key}_{zone}"] = float(data.max())
                        overall_metrics[f"min_{key}_{zone}"] = float(data[data != 0].min()) if data.sum() != 0 else 0
                    else:
                        overall_metrics[f"mean_{key}_{zone}"] = float(data[data != 0].mean())
                        overall_metrics[f"max_{key}_{zone}"] = float(data[data != 0].max())
                        overall_metrics[f"min_{key}_{zone}"] = float(data[data != 0].min())
        else:
            interp_size = chunk_size // self.frames_to_interpolate  # chunk size inside section
            for key in self.metrics.keys():
                for i, zone in enumerate(('A', 'B', 'C', 'D')):
                    data = np.array(self.metrics[key][i * chunk_size : (i + 1) * chunk_size])  # metric inside zone
                    if key == 'total_spatters':
                        overall_metrics[f"{key}_{zone}"] = []
                    else:
                        overall_metrics[f"mean_{key}_{zone}"] = []
                        overall_metrics[f"max_{key}_{zone}"] = []
                        overall_metrics[f"min_{key}_{zone}"] = []
                    for j in range(interp_size):
                        chunk = data[j * self.frames_to_interpolate : (j + 1) * self.frames_to_interpolate]
                        if key == 'total_spatters':
                            overall_metrics[f"{key}_{zone}"].append(int(chunk.max() - chunk.min()))
                        elif key in ('appearance_rate', 'n_spatters'):
                            overall_metrics[f"mean_{key}_{zone}"].append(float(chunk.mean()))
                            overall_metrics[f"max_{key}_{zone}"].append(float(chunk.max()))
                            overall_metrics[f"min_{key}_{zone}"].append(float(chunk[chunk != 0].min()) if chunk.sum() != 0 else 0)
                        else:
                            overall_metrics[f"mean_{key}_{zone}"].append(float(chunk[chunk != 0].mean()) if chunk.sum() != 0 else 0)
                            overall_metrics[f"max_{key}_{zone}"].append(float(chunk[chunk != 0].max()) if chunk.sum() != 0 else 0)
                            overall_metrics[f"min_{key}_{zone}"].append(float(chunk[chunk != 0].min()) if chunk.sum() != 0 else 0)

        return overall_metrics

    def append(self, tracks: Dict[int, List[List[np.ndarray]]], t_frame: np.ndarray, center: np.ndarray) -> None:
        """
    Processes the tracks and frame data to calculate various metrics related to spatters such as welding zone temperature, 
    total number of spatters, mean size, temperature, cooling speed and velocity of spatters per frame, and new spatter
    appearance rate.

    Args:
        self (object): The instance of the class.
        tracks (Dict[int, List[List[np.ndarray]]]): A dictionary where each key is an integer and the value is a list of numpy arrays.
            Each numpy array represents a track.
        t_frame (np.ndarray): A numpy array representing the current frame.
        center (np.ndarray): A numpy array representing the center of the current frame.

    Returns:
        None
    """
        if len(tracks):
            # Temperature of welding zone (max)
            x, y, w, h = center
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            self.metrics['welding_zone_temp'].append(t_frame[y1:y2, x1:x2].max())
            # Total number of spatters
            self.metrics['total_spatters'].append(max(list(tracks.keys())))  # compute accumulated number of spatters
            # Number of spatters per frame
            self.metrics['n_spatters'].append(len(tracks))
            # Mean size of spatters per frame
            pts = np.array(list(tracks.values()))  # collect bboxes of all kpts on the previous 3 frames (shape: )
            cur_pts = pts[:, 0, :]  # get kpts on the current frame
            areas = cur_pts[:, 2] * cur_pts[:, 3]  # save size of current spatters
            self.metrics['size'].append(areas.mean())
            # Mean temperature of spatters per frame
            # Mean cooling speed of spatters per frame
            temp = []
            cool_speed = []
            for key, value in tracks.items():
                bboxes = np.array(value.copy())
                bboxes[:2, :2] = bboxes[:2,:2] - bboxes[:2,2:4] / 2
                bboxes[:2,2:4] = bboxes[:2,2:4] + bboxes[:2,:2]
                bboxes = bboxes.astype(int)
                x1, y1, x2, y2 = bboxes[0]
                cur_temp = t_frame[y1:y2, x1:x2].mean()
                temp.append(cur_temp)
                x1, y1, x2, y2 = bboxes[1]
                prev_temp = self.last_t_frame[y1:y2, x1:x2].mean()
                cool_speed.append(np.abs(prev_temp - cur_temp))
            
            self.metrics['temp'].append(np.array(temp).mean())
            self.metrics['cooling_speed'].append(np.array(cool_speed).mean())
            # Mean velocity of spatters per frame
            vel = np.sqrt(np.square(pts[:, 0, :2] - pts[:, 1, :2]).sum(axis=-1)).mean()
            self.metrics['velocity'].append(vel)
            # Number of new spatters per frame (appearance rate)
            if self.last_tracks is not None:
                new_ids = set(tracks.keys()) - set(self.last_tracks.keys())
                self.metrics['appearance_rate'].append(max(0, len(new_ids)))
            else:
                self.metrics['appearance_rate'].append(0)
            self.last_tracks = tracks.copy()
        else:
            for key in self.metrics.keys():
                self.metrics[key].append(0)
            self.last_tracks = None

        self.last_t_frame = t_frame  # save for the next iteration
        #print(self.metrics)
             

def process_thermogram(path: str, w_size: int) -> int:
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

    tracker = Tracker()

    scale = 2
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #recorder = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 512))

    fe = FeatureExtractor(w_size)

    k = 0

    for frame, t_frame in zip(frames, temp_frames):
        #print(k)
        k += 1
        pts = np.zeros((frame.shape[0] * scale, frame.shape[1] * scale) + (3,))  # for drawing

        try:
            center, ellips = detect_welding_zone(frame)
            frame = cv2.ellipse(frame, ellips, (255,), 1)
        except:
            continue

        boxes = detect_spatters(frame)
        try:
            boxes = filter_spatters(boxes, center, R)
            #boxes = remove_reflection(boxes, center, DX_REFL, DY_REFL, R_REFL)
            #boxes = remove_reflection(boxes, center, DX_TR, DY_TR, R_TR)
            x, y, w, h = center * scale
            pts = cv2.circle(pts, (int(x), int(y)), R * scale, (130, 130, 130), 1)
            #pts = cv2.circle(pts, (int(x + DX_REFL * scale), int(y + DY_REFL * scale)), R_REFL * scale, (255, 0, 0), 1)
            #frame = cv2.circle(frame, (int(x / scale + DX_REFL), int(y / scale + DY_REFL)), R_REFL, (255, ), 1)
            #pts = cv2.circle(pts, (int(x + DX_TR * scale), int(y + DY_TR * scale)), R_TR * scale, (0, 255, 0), 1)
            #frame = cv2.circle(frame, (int(x / scale + DX_TR), int(y / scale + DY_TR)), R_TR, (255, ), 1)
        except:
            pass

        tracked = tracker.step(boxes)

        fe.append(tracked, t_frame, center)

        # for pt in boxes:
        #     x, y, w, h = pt * scale
        #     pts = cv2.circle(pts, (int(x), int(y)), int((w + h) / 2), (0, 0, 255), 1)    
        

        if len(tracked) > 0:
            for track in tracked.values():
                pt1, pt2, pt3 = track
                x1, y1 = pt1[:2] * scale
                x2, y2 = pt2[:2] * scale
                x3, y3 = pt3[:2] * scale
                pts = cv2.line(pts, (x1, y1), (x2, y2), (255, 255, 255), 1)
                pts = cv2.line(pts, (x3, y3), (x2, y2), (255, 255, 255), 1)
                pts = cv2.circle(pts, (int(x3), int(y3)), 4, (0, 0, 255), 1)


        pts = cv2.putText(pts, f"Number of spatters {tracker.max_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 1)
        
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        frame = np.concatenate((frame, pts), axis=1).astype(np.uint8)
        # cv2.imshow('thermogram', frame)
        # if cv2.waitKey(0) == ord('q'):
        #    break
        #recorder.write(frame)

    #recorder.release()
    #cv2.destroyAllWindows()
    #print(fe.compute_results())
    return fe.compute_results(True).copy()

import os

W_SIZE = 10

paths = os.listdir('data/')
counts = {}

for path in tqdm(paths):
   counts[path] = process_thermogram(f"data/{path}", W_SIZE)

#process_thermogram('data/thermogram_11.npy', 10)

import json

with open(f"metrics_{W_SIZE}.json", 'w') as f:
    json.dump(counts, f)

# print(counts)
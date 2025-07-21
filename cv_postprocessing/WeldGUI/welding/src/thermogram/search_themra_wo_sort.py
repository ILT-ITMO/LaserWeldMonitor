import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics as s
from SORT import Sort


for exp in [7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26]:
    print(exp)
    #tracker = Sort(max_age=3, min_hits=1, iou_threshold=0)
    arr = np.load(fr'D:\Projects\Transformers\svarka\data\{exp}_export.npy')
    # for i in arr:
    summ_count = []
    total_fraim_points = []
    max_ =[]
    for frame in arr:
        #gray = ((arr[512]>500)&(arr[512]<600)).astype(int)
        gray0 = ((frame>250)).astype(int)
        #     plt.imshow(i)
        #     plt.show()
        
        #gray0 =  cv2.GaussianBlur(gray0.astype(np.uint8),(3,3),0)
        c,_ = cv2.findContours((gray0*255).astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for_dect = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3],1]) for i,_ in enumerate(c)]
        # if len(for_dect)>0:
        #     id_detect = tracker.update(np.array(for_dect))
        #     if len(id_detect)>0:
        #         if id_detect[:,-1].max()>max_track:
        #             max_track=id_detect[:,-1].max()
        # else:
        #     id_detect = tracker.update(np.empty((0, 5)))
        areas = []
        for ci in c:
            areas.append(cv2.contourArea(ci))
        # if len(areas)==0:
        #     summ_count.append(0)
        # else:
        if len(areas)==0:
            summ_count.append(0)
        else:
            #max_a = max(areas)//2
            summ_count.append(areas)
            if areas!=0.0 or max(areas)!=0.0:
                 max_.append(max(areas))
    # for frame_areas in summ_count:
    #     if frame_areas==0:
    #         continue
    #     if max(frame_areas)==0.0:
    #         continue
    #     max_.append(max(frame_areas))
    for frame_areas in summ_count:
        if frame_areas==0:
            continue
        total_fraim_points.append(sum(np.array(frame_areas)[np.array(frame_areas)<s.mean(max_)]))
    print(max(total_fraim_points),sum(total_fraim_points),sum(total_fraim_points)/len(total_fraim_points))#summ_count,
# print(summ_count[254])
# gray0 = (arr[254]>0).astype(int)
# #gray0 =  cv2.medianBlur(gray0.astype(np.uint8),3)
# plt.imshow(gray0)
# plt.show()
# plt.imshow(cv2.GaussianBlur(arr[512].astype(np.uint8),(3,3),0))
# plt.show()
# print()
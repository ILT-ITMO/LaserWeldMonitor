import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics as s
from SORT import Sort
import time
# def animation_npz(path_to_dir):
    
#     cnt = 0
#     plt.ion()
#     fig, axs = plt.subplots(1,3, figsize=(15, 10))
#     summ_mae = 0
#     for file in l_dir:
#         mask = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow_predict']#mask
#         flow = np.load(path_to_dir+f'/{file}',allow_pickle = True)['arr_0'].item(0)['flow']
#         for m,f in zip(mask,flow):
#             cnt+=1
#             mse = np.square(f-np.where(m>-0.01,m,-1)).mean()
#             mae = np.sum(np.absolute(f-np.where(m>-0.01,m,-1)))/400/400
#             summ_mae+=np.sum(np.absolute(np.where(f>0,f,0)-np.where(m>-0.1,m,0)))
#             m,f = np.where(m>-0.1,m,np.inf),np.where(f>0,f,np.inf)
#             a1 = axs[0].imshow(m, vmin = 0, vmax = 0.05)
            
#             axs[1].set_title(f'summMAE: {summ_mae}')
#             axs[2].set_title(f'MAE: {mae},MSE: {mse}')
#             a2 =axs[1].imshow(f, vmin = 0, vmax = 0.05)
             
#             a3 = axs[2].imshow(np.absolute(f-m),vmin = 0, vmax = 0.007)
            
#             # fig.colorbar(a1,ax=axs[0])
#             # fig.colorbar(a2,ax=axs[1])
#             # fig.colorbar(a3,ax=axs[2])
            
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#             axs[0].clear()
#             axs[1].clear()
#             axs[2].clear()
            
            
#             #time.sleep(2.2)
#             a1.remove(),a2.remove(),a3.remove()
#         print(summ_mae)
#         print(cnt)


# #animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim')
# #animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim_23')
# #animation_npz(path_to_dir='gefest\surrogate_models\gendata/ssim_plus_57')
# #animation_data_npz(path_to_dir='data_from_comsol/gen_data_extend')
# #animation_npz(path_to_dir='gefest\surrogate_models\gendata/unet_68_Adam_att_unet_bi_wu_f01_t01_10ep_bs32_30_to_0001_ssim_2')
# animation_npz(path_to_dir=r'D:\Projects\GEFEST\GEFEST_surr\GEFEST\gefest\surrogate_models\gendata\unet_69_Adam_from01_lr20to3e104_bs4')


tracker = Sort(max_age=3, min_hits=1, iou_threshold=0)
for exp in [14]:#[7,8,9,10,11,12,13,14]:
    arr = np.load(fr'D:\Projects\Transformers\svarka\data\{exp}_export.npy')
    # for i in arr:
    summ_count = []
    plt.ion()
    fig, axs = plt.subplots(1,1, figsize=(15, 10))
    total_fraim_points = []
    max_ =[]
    for frame in arr:
        gray0 = ((frame>200)).astype(int)
        #     plt.imshow(i)
        #     plt.show()
        
        #gray0 =  cv2.GaussianBlur(gray0.astype(np.uint8),(3,3),0)
        c,_ = cv2.findContours((gray0).astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for_dect = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3],1]) for i,_ in enumerate(c)]
        if len(for_dect)>0:
            id_detect = tracker.update(np.array(for_dect))
        else:
            id_detect = tracker.update(np.empty((0, 5)))
        for i in id_detect:
            plt.plot([i[0],i[2]],[i[1],i[3]])
        #plt.plot(frame)
        #a2 =axs.imshow(((frame>200)&(frame<frame.max()//1.5)).astype(int))
        a2 =axs.imshow(gray0)
        
        #plt.plot(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.3)
        axs.clear()
        a2.remove()
        #gray = ((arr[512]>500)&(arr[512]<600)).astype(int)
        # gray0 = (frame>0).astype(int)
        # #     plt.imshow(i)
        # #     plt.show()
        
        # #gray0 =  cv2.GaussianBlur(gray0.astype(np.uint8),(3,3),0)
        # c,_ = cv2.findContours((gray0*255).astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # areas = []
        # for ci in c:
        #     areas.append(cv2.contourArea(ci))
        # # if len(areas)==0:
        # #     summ_count.append(0)
        # # else:
        # if len(areas)==0:
        #     summ_count.append(0)
        # else:
        #     #max_a = max(areas)//2
        #     summ_count.append(areas)
        #     if areas!=0.0 or max(areas)!=0.0:
        #          max_.append(max(areas))
    # for frame_areas in summ_count:
    #     if frame_areas==0:
    #         continue
    #     if max(frame_areas)==0.0:
    #         continue
    #     max_.append(max(frame_areas))

    # for frame_areas in summ_count:
    #     if frame_areas==0:
    #         continue
    #     total_fraim_points.append(sum(np.array(frame_areas)[np.array(frame_areas)<s.mean(max_)]))
    # print(max(total_fraim_points),sum(total_fraim_points),sum(total_fraim_points)/len(total_fraim_points))#summ_count,
# print(summ_count[254])
# gray0 = (arr[254]>0).astype(int)
# #gray0 =  cv2.medianBlur(gray0.astype(np.uint8),3)
# plt.imshow(gray0)
# plt.show()
# plt.imshow(cv2.GaussianBlur(arr[512].astype(np.uint8),(3,3),0))
# plt.show()
# print()
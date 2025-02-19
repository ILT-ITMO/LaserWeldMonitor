import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
from scipy.spatial import distance

def crop(mask,stride=10):
    y,x = np.where(mask==255)
    y_min,y_max,x_min,x_max = y.min(),y.max(),x.min(),x.max()
    return mask[y_min-stride:y_max+stride,x_min-stride:x_max+stride]

def diag(matrix):
    ind_i = 0
    for i in matrix:
        ind_j = 0
        for j in i:
            if ind_i==ind_j:
                if j==255:
                    return (ind_i,ind_j)
                else:
                    ind_j+=1
            else:
                ind_j+=1
        ind_i+=1

def diag_45(matrix:np.array,from_=1):
    """Function search coordinates from angels of weld mask.
    The search is carried out by drawing a tangent from the corner of the picture selected in the FROM_ quarter.


    Args:
        matrix (np.array): Matrix with mask of weld
        from_ (int, ): Selecting a quarter of the mask coordinates in which the point is searched. Defaults to 1.
        [1       2]
        [3       4]

    Returns:
        List[int]: The coordinates of the found points [x,y] in the quadrant are returned.
    """
    if from_==1:
        #from left top
        for i in range(matrix.shape[1]):
            j=0
            while j!=i and j<matrix.shape[0]:
                if matrix[j,i-j] == 255:
                    return (j,i-j)
                else:
                    j+=1
    if from_==2:
        #from rights top
        for i in range(matrix.shape[1]-1,0,-1):
            j=0
            while i+j<matrix.shape[1] and j<matrix.shape[0]:
                if matrix[j,i+j] == 255:
                    return (j,i+j)
                else:
                    j+=1
    if from_==3:
        #from rights botttom
        for i in range(matrix.shape[1]-1,0,-1):
            j=matrix.shape[0]-1
            cnt_i=0
            cnt_j=0
            while i+cnt_i<matrix.shape[1] and j-cnt_j>=0:
                if matrix[j-cnt_j,i+cnt_i] == 255:
                    return (j-cnt_j,i+cnt_i)
                else:
                    cnt_i+=1
                    cnt_j+=1
    
    if from_==4:
        #from left botttom
        for i in range(matrix.shape[1]):
            j=matrix.shape[0]-1
            cnt_i=0
            cnt_j=0
            while i-cnt_i>=0 and j-cnt_j>=0:
                if matrix[j-cnt_j,i-cnt_i] == 255:
                    return (j-cnt_j,i-cnt_i)
                else:
                    cnt_i+=1
                    cnt_j+=1
    
def plot_mask_and_point(path):
    img = Image.open(path)
    mask = np.array( img, dtype='uint8' )
    #mask = np.load(r'project-8-at-2024-04-01-13-48-4991592d\task-1-annotation-4-by-1-tag-Weld-0.npy')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    top_left = diag_45(crop(mask))
    top_right = diag_45(crop(mask),from_=2)
    coords3 = diag_45(crop(mask),from_=3)
    coords4 = diag_45(crop(mask),from_=4)
    y,x = np.where(mask==255)
    ax.imshow(crop(mask))
    plt.plot([top_left[1],top_right[1]],[top_left[0],top_right[0]],label='top line')
    plt.plot([coords3[1],coords4[1]],[coords3[0],coords4[0]],label='bottom line')
    plt.legend()
    plt.scatter(top_left[1],top_left[0])
    plt.scatter(top_right[1],top_right[0])
    plt.scatter(coords3[1],coords3[0])
    plt.scatter(coords4[1],coords4[0])
    coord = diag(crop(mask))

    plt.show()

def return_points_and_size(path):
    img = Image.open(path)
    mask = np.array( img, dtype='uint8' )
    #mask = np.load(r'project-8-at-2024-04-01-13-48-4991592d\task-1-annotation-4-by-1-tag-Weld-0.npy')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    top_left = diag_45(crop(mask))
    top_right = diag_45(crop(mask),from_=2)
    bot_left = diag_45(crop(mask),from_=3)
    bot_right = diag_45(crop(mask),from_=4)
    top_coords = [top_left[1],top_left[0]],[top_right[1],top_right[0]]
    bot_coords = [bot_left[1],bot_left[0]],[bot_right[1],bot_right[0]]
    top_line_len = distance.euclidean(*top_coords)
    bot_line_len = distance.euclidean(*bot_coords)

    return top_line_len,top_coords, bot_line_len,bot_coords

if __name__=='__main__':
    #return_points_and_size(path)
    img = Image.open(r'D:\Projects\weld\weldseg\masks\00860-3569846766.png')
    mask = np.array( img, dtype='uint8' )
    #mask = np.load(r'project-8-at-2024-04-01-13-48-4991592d\task-1-annotation-4-by-1-tag-Weld-0.npy')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    top_left = diag_45(crop(mask))
    top_right = diag_45(crop(mask),from_=2)
    coords3 = diag_45(crop(mask),from_=3)
    coords4 = diag_45(crop(mask),from_=4)
    y,x = np.where(mask==255)
    ax.imshow(crop(mask))
    plt.plot([top_left[1],top_right[1]],[top_left[0],top_right[0]],label='top line = ')
    plt.plot([coords3[1],coords4[1]],[coords3[0],coords4[0]],label='bottom line')
    plt.legend()
    plt.scatter(top_left[1],top_left[0])
    plt.scatter(top_right[1],top_right[0])
    plt.scatter(coords3[1],coords3[0])
    plt.scatter(coords4[1],coords4[0])
    coord = diag(crop(mask))

    plt.show()
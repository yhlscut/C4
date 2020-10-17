import os

import cv2
import numpy as np

BOARD_FILL_COLOR = 1e-5

def main():
    datadir='./data/images/'
    list_path = './data/color_cheker_data_meta.txt'
    with open(list_path,'r') as f:
        all_data_list = f.readlines()
    # make dir
    if not os.path.exists('./data/ndata/'):
        os.mkdir('./data/ndata')
    if not os.path.exists('./data/nlabel'):
        os.mkdir('./data/nlabel')
    # generate npy data
    for l in all_data_list:
        illums = []
        file_name = l.strip().split(' ')[1]
        illums.append(float(l.strip().split(' ')[2]))
        illums.append(float(l.strip().split(' ')[3]))
        illums.append(float(l.strip().split(' ')[4]))
        np.vstack(illums)
        mcc_coord = get_mcc_coord(file_name)
        img_without_mcc = load_image_without_mcc(file_name,mcc_coord)
        np.save('./data/ndata/'+file_name+'.npy',img_without_mcc) #img BGR
        np.save('./data/nlabel/'+file_name+'.npy',illums)
        print(file_name)

def load_image_without_mcc(file_name,mcc_coord):
    raw = load_image(file_name)
    img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.float32) # clip constrain the value between 0 and 1
    polygon = mcc_coord * np.array([img.shape[1], img.shape[0]]) #the vertex of polygon
    polygon = polygon.astype(np.int32) 
    cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3) # fill the polygon to img
    return img 
        
def load_image(file_name):
    file_path = './data/images/' + file_name
    raw = np.array(cv2.imread(file_path, -1), dtype='float32')
    if file_name.startswith('IMG'):
      # 5D3 images
      black_point = 129
    else:
      black_point = 1
    raw = np.maximum(raw - black_point, [0, 0, 0])  # remain the pixels that raw-black_point>0
    return raw      

def get_mcc_coord(file_name):
    # Note: relative coord
    with open('./data/coordinates/' + file_name.split('.')[0] +'_macbeth.txt', 'r') as f:
        lines = f.readlines()
        width, height = map(float, lines[0].split())
        scale_x = 1 / width
        scale_y = 1 / height
        lines = [lines[1], lines[2], lines[4], lines[3]]
        polygon = []
        for line in lines:
            line = line.strip().split()
            x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
            polygon.append((x, y))
        return np.array(polygon, dtype='float32') 
        
if __name__=='__main__':
    main()        

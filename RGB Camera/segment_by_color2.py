import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from PIL import Image
def get_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))
def rgb2hsv(r,g,b):
    v = max(r,g,b)
    s = 0 if v == 0 else (v-min(r,g,b)) / v
    if v == r:
        h = 60 * (g - b) / (v - min(r,g,b))
    elif v == g:
        h = 120 + 60*(b-r)/(v - min(r,g,b))
    else:
        h = 240 + 60*(r-g)/(v-min(r,g,b))
    return h / 2,s * 255,v

def seg_by_hsv(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h,s,v = img_hsv.transpose([2,0,1])

    mask1 = (h <= target_hsv[0]*1.25) & (h >= target_hsv[0]*0.75)
    mask2 = s > 50
    mask3 = v >= 63.75
    mask = mask1 & mask2 & mask3

    return mask

def seg_by_rgb(img):
    rgb = img.transpose([2,0,1])[::-1, :, :]
    rgb_vec_tg = np.array(target_rgb)
    I = rgb - rgb_vec_tg.reshape(3, 1, 1)
    mask = (abs(I[0] - I[1]) < 25) & (abs(I[1] - I[2]) < 25)
    return mask


    


video_path = os.path.join(os.path.dirname(__file__), r'resource\v1.mp4')
target_rgb = 96, 128, 149
target_hsv = rgb2hsv(*target_rgb)

cap = cv.VideoCapture(video_path)
# cap = cv.VideoCapture(0)
# print(video_path)

while 1:
    ok, img = cap.read()
    if not ok: break

    # === Manipulate Image ===
    # img = cv.pyrDown(img) # 缩小
    # img = cv.equalizeHist(img)
    

    # --- Calculate Mask ---
    mask_hsv = seg_by_hsv(img)
    mask_rgb = seg_by_rgb(img)
    mask = mask_hsv & mask_rgb

    # --- Remove Noise ---
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_ERODE, kernel)

    # --- Find and Draw Contours ---
    contours, arch  = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()
    cv.drawContours(img_contours, contours, -1, (255,0,0), thickness=2)

    # --- Find and Draw Bounding Box ---
    area = np.zeros(len(contours))
    max_area = img.shape[0] * img.shape[1]
    prop = np.zeros(len(contours))
    pss = []
    for i, c in enumerate(contours):
        ps = cv.boxPoints(cv.minAreaRect(c))
        pss.append(ps)
        area[i] = get_distance(ps[0], ps[1]) * get_distance(ps[1], ps[2])
        prop[i] = get_distance(ps[0], ps[1]) / (get_distance(ps[1], ps[2]) + 1e-10)

    

    img_rect = img.copy()
    # cv.putText(img_rect, f'Mean S(%):{s.mean()/255:.1f},Mean V(%):{v.mean()/255:.1f}', (0, img_rect.shape[0]-1), 
        # cv.FONT_HERSHEY_SIMPLEX,
        # 1.3,
        # (255,255,255), 3)
    
    # Only show largest 5
    for idx in area.argsort()[-5:]:
        
        if prop[idx] > 3 or prop[idx] < 0.33 or area[idx]/ max_area < 0.02:
            continue
        p = tuple(pss[idx][0])
        cv.putText(img_rect, f'Area:{area[idx]/max_area:.1f},Prop:{prop[idx]:.1f}', p, 
        cv.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0,0,0), 3)
        for i in range(4):
            cv.line(img_rect, tuple(pss[idx][i]), tuple(pss[idx][(i + 1) % 4]), (0,0,255), 2)

    # === Show Video ===
    cv.imshow("Video", cv.pyrDown(img))
    cv.imshow("Mask", cv.pyrDown(mask*255))
    cv.imshow("Mask HSV", cv.pyrDown(mask_hsv.astype(np.uint8)*255))
    cv.imshow("Mask RGB", cv.pyrDown(mask_rgb.astype(np.uint8)*255))
    cv.imshow("Contours", cv.pyrDown(img_contours))
    cv.imshow("Bounding Box", cv.pyrDown(img_rect))
    key_in = cv.waitKey(33)
    if key_in != -1:
        if key_in == 32:
            cv.waitKey(0)
            continue
        else:
            break

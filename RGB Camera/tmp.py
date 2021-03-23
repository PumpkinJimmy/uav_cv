import cv2 as cv
import time

cap = cv.VideoCapture(0)


while 1:
    ok, img = cap.read()
    if not ok: break
    cv.imshow('Camera', img)
    key_in = cv.waitKey(33)
    if key_in != -1:
        if key_in == 32:
            img_name = f'{time.time():.0f}.jpg'
            cv.imwrite(img_name, img)
            print(f"Saved at {img_name}")
        else:
            break
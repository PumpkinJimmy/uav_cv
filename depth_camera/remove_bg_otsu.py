#####################################################
# Otsu自适应二值化
# 假设空旷的平地上只有目标物体，则本方法可以有效分割出背景
# 问题在于有干扰的时候做不了
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

def seg_by_hsv(img, target_hsv=(101.89, 90.7, 149.0), hmin=75, hmax=125, st=20, vt=25):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v = img_hsv.transpose([2,0,1])

    mask1 = (h <= target_hsv[0]*(hmax / 100)) & (h >= target_hsv[0]*(hmin/100))
    mask2 = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask3 = v >= 255 * vt / 100
    mask = mask1 & mask2 & mask3
    return mask

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        thresh, mask = cv2.threshold(depth_image, 0, 255, cv2.THRESH_OTSU)
        mask = ~(mask.astype(bool))
        mask &= (depth_image!=0)
        cv2.imshow("Otus Mask", mask.astype(np.uint8)*255)
        mask_hsv = seg_by_hsv(color_image)
        cv2.imshow('Color Mask', mask_hsv * 255)
        bg_removed = np.where(np.dstack([mask&mask_hsv]*3), color_image, grey_color)
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0) | ~mask_hsv, grey_color, color_image)

        

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
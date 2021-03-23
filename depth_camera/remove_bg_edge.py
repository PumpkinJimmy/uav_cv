#####################################################
# 基于边缘的分割
# 好处是哪怕物体有多个也可以跑
# 问题是噪声太多了，基本上找不到目标
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

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

        # =========== body =============
        mask_dx = cv2.Sobel(depth_image, -1, 1, 0)
        # cv2.imshow('Sobel_dx', mask_dx)
        mask_dy = cv2.Sobel(depth_image, -1, 0, 1)
        # cv2.imshow('Sobel_dy', mask_dy)
        mask_sobel = cv2.convertScaleAbs(mask_dx * 0.5 + mask_dy * 0.5)
        cv2.imshow('Sobel', mask_sobel)

        mask_sobel2 = cv2.threshold(mask_sobel, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('Sobel2', mask_sobel2)

        depth_image_8u = cv2.convertScaleAbs(depth_image)
        cv2.imshow('Depth_8U', depth_image_8u)
        mask_canny = cv2.Canny(depth_image_8u, 50, 255)
        # mask_canny = cv2.Canny(mask_dx.astype(np.int16), mask_dy.astype(np.int16), 0, 30000)
        cv2.imshow("Canny", mask_canny)

        # =========== end body =============

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
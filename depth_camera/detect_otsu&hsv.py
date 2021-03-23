import pyrealsense2 as rs
import numpy as np
import cv2 as cv

def seg_by_hsv(img_hsv, 
    target_hsv=(101.89, 90.7, 149.0),
    tol={
        'hmin':75,
        'hmax':125,
        'st':20,
        'vt':25
    }):
    '''
    Segment the image by HSV color

    img_hsv: np.array (h, w, 3), HSV Image
    target_hsv: tuple, target hsv value
        H: [0, 180]
        S: [0, 255]
        V: [0, 255]
    tol: tolerance of each value (by percentage)
        H: hmin, hmax
        S: ignored
        V: vt
    return: mask: np.array (h, w)
    '''
    h,s,v = img_hsv.transpose([2,0,1])
    hmin = tol['hmin']
    hmax = tol['hmax']
    st = tol['st']
    vt = tol['vt']

    mask1 = (h <= target_hsv[0]*(hmax / 100)) & (h >= target_hsv[0]*(hmin/100))

    mask2 = cv.threshold(s, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    mask3 = v >= 255 * vt / 100

    mask = mask1 & mask2 & mask3
    return mask

def seg_by_otsu(img_depth, ksize=7):
    '''
    Segment image by depth with OTSU threshold

    img_depth: np.array(dtype=uint16), depth image

    ksize: kernel size for dilate operation

    return: mask
    '''
    thresh, mask_otsu = cv.threshold(img_depth, 0, 255, cv.THRESH_OTSU)
    mask_otsu = ~(mask_otsu.astype(bool))
    mask_otsu &= (img_depth!=0)
    mask_otsu = cv.morphologyEx(mask_otsu.astype(np.uint8), cv.MORPH_DILATE, np.ones((ksize,ksize)))
    return mask_otsu

def init_rs_pipeline():
    '''
    Initialize the RealSense Pipeline and align depth to color
    '''
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

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, depth_scale, align

def find_bounding_box(mask, prop_lim=3):
    '''
    Find the bounding box from the mask

    mask: np.array, image mask

    prop_lim: limitation of H/W or W/H proportion

    return: tuple, tuple of points represent rotated rect, or None

    '''
    def get_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    contours, arch  = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    # Find the Bounding Box
    area = np.zeros(len(contours))
    prop = np.zeros(len(contours))
    pss = []

    # Calculate the area and proportion
    for i, c in enumerate(contours):
        ps = cv.boxPoints(cv.minAreaRect(c))
        pss.append(ps)
        area[i] = get_distance(ps[0], ps[1]) * get_distance(ps[1], ps[2])
        prop[i] = get_distance(ps[0], ps[1]) / (get_distance(ps[1], ps[2]) + 1e-10)
    

    # Filter and select the target
    for idx in area.argsort()[-1:]:
        if prop[idx] > prop_lim or prop[idx] < 1/prop_lim:
            continue
        return pss[idx]

def get_center(rrect):
    p1 = rrect[0]
    p2 = rrect[2]
    return int((p1[0] + p2[0]) // 2), int((p1[1] + p2[1]) // 2)

# ======= main =============

pipeline, depth_scale, align = init_rs_pipeline()

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
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels

        mask_otsu = seg_by_otsu(depth_image)

        cv.imshow("Otus Mask", mask_otsu.astype(np.uint8)*255)

        mask_hsv = seg_by_hsv(cv.cvtColor(color_image, cv.COLOR_BGR2HSV))
        cv.imshow('Color Mask', mask_hsv * 255)

        # mask = mask_otsu & mask_hsv
        mask = mask_hsv & mask_otsu
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3)))
        cv.imshow('Mask', mask * 255)


        img_contours = color_image.copy()

        # --- Find and Draw Bounding Box ---
        rrect = find_bounding_box(mask)
        
        if rrect is not None:
            brect = cv.boundingRect(rrect)
            cv.rectangle(img_contours, brect, (0,0,255), 2)
            center = get_center(rrect)
            cv.putText(img_contours, '{:.2f}'.format(depth_image[center[1], center[0]]*depth_scale), center,
            cv.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0,0,0), 3)
        
        # Render images
        bg_removed = np.where(np.dstack([mask]*3), color_image, grey_color)

        cv.imshow("Bounding Box", img_contours)

        
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        # images = bg_removed
        cv.namedWindow('Align Example', cv.WINDOW_AUTOSIZE)
        cv.imshow('Align Example', images)
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            break
finally:
    pipeline.stop()


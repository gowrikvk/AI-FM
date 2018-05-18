import numpy as np
import cv2
import imutils
class FeatureImageDetector:

    def __init__(imagedata):
        # Feature-point detector
        imagedata.feature_detector = cv2.AKAZE_create()
        # ROI (Region-Of-Interest) to learn a target object
        imagedata.sub_topleft = [100, 220]
        imagedata.sub_width = 200
        imagedata.sub_height = 200
        imagedata.sub_bottomright = [imagedata.sub_topleft[0] + imagedata.sub_height - 1,\
                                imagedata.sub_topleft[1] + imagedata.sub_width - 1]
        imagedata.rect_color = (0, 255, 200)
        imagedata.rect_thickness = 3
        imagedata.rect_tl_outer_xy = (imagedata.sub_topleft[1] - imagedata.rect_thickness,\
                                 imagedata.sub_topleft[0] - imagedata.rect_thickness)
        imagedata.rect_br_outer_xy = (imagedata.sub_bottomright[1] + imagedata.rect_thickness,\
                                 imagedata.sub_bottomright[0] + imagedata.rect_thickness)
        # Threshold for the distance of feature (descriptor) vectors
        imagedata.ratio = 0.75
        imagedata.registered = False
        imagedata.min_match_count = 4

    def registerFeature(imagedata):
        temp = True
        if temp == True:
            #frame = cv2.imread('dollar.png')
            frame = cv2.imread('doll.jpg')
            frame = imutils.resize(frame)
            # cv2.rectangle(frame, imagedata.rect_tl_outer_xy, imagedata.rect_br_outer_xy,\
            #               imagedata.rect_color, imagedata.rect_thickness)
            imagedata.kp0, imagedata.des0 = imagedata.feature_detector.detectAndCompute(frame, None)
            imagedata.querying = frame
            imagedata.registered = True

    def detectFeature(imagedata):
        bf = cv2.BFMatcher()
        temp = True
        frame = cv2.imread('box_in_scene.jpg')
        frame = cv2.imread('chinchq.jpg')
        frame = imutils.resize(frame)
        bkp_frame = cv2.imread('chinchq.jpg')
        # Keypoint (kp) detection and calculate descriptors (des)
        kp, des = imagedata.feature_detector.detectAndCompute(frame, None)
        winframe=imutils.resize(frame,1000,1000)
        cv2.imshow(" Actual Img", winframe)
        cv2.waitKey(0)
        cv2.imshow(" Feature Image : ", imagedata.querying)
        cv2.waitKey(0)
        # Apply blute-force knn matching between keypoints
        matches = bf.knnMatch(imagedata.des0, des, k=2)
        # Adopt only good feature matches
        goodfeatures = [[m] for m, n in matches if m.distance < imagedata.ratio * n.distance]
        print(len(goodfeatures))
        #Find Homography
        if len(goodfeatures) > imagedata.min_match_count:
            src_pts = np.float32([imagedata.kp0[m[0].queryIdx].pt for m in goodfeatures]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m[0].trainIdx].pt for m in goodfeatures]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w, _ = imagedata.querying.shape  # Assume color camera
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            print(type(M))
            print(len(M))
            dst = cv2.perspectiveTransform(pts, M)

            for i, coords in enumerate(dst):
                if i == 0:
                    x1, y1 = coords[0]
                elif i == 1:
                    x2, y2 = coords[0]
                elif i == 2:
                    x3, y3 = coords[0]
                else:
                    x4, y4 = coords[0]
            cropped_img = bkp_frame[int(y1):int(y3), int(x2):]
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
        #Visualize the matches
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0)
        img = cv2.drawMatchesKnn(imagedata.querying, imagedata.kp0, frame, kp, goodfeatures, None, **draw_params)
        img=imutils.resize(img, 1000,1000)
        cv2.imshow("feature match : ",img)
        cv2.waitKey()
        cv2.imshow("Amount portion : ",cropped_img)
        cv2.waitKey()

    def closeWindows(imagedata):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    objDetector = FeatureImageDetector()
    objDetector.registerFeature()
    objDetector.detectFeature()
    objDetector.closeWindows()

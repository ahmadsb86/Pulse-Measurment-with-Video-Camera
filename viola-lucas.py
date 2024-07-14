import os
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# Init stuff
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(1)

color = np.random.randint(0, 255, (100, 3)) 

# params for corner detection 
feature_params = dict( maxCorners = 100, 
					qualityLevel = 0.05, 
					minDistance = 7, 
					blockSize = 7 ) 

lk_params = dict( winSize = (15, 15), 
				maxLevel = 2, 
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
							10, 0.03)) 

#Stage 1: Detect Face
# x and y represent coordinates of the 4 corners of the face bounding box
cx = np.zeros(4)
cy = np.zeros(4)
faceFound = False
while not faceFound:
    ret, old_frame = video_capture.read()
    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(old_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if(len(faces)>0):
        x,y,w,h = faces[0]
        cx[0],cy[0] = x,y
        cx[1],cy[1] = x+w,y
        cx[2],cy[2] = x+w,y+h
        cx[3],cy[3] = x,y+h
        
        faceFound = True


#Stage 2: Find tracking points
facemask = np.zeros_like(old_gray)
facemask = cv2.rectangle(facemask, (x, y), (x+w, y+h),255,-1)
old_points = cv2.goodFeaturesToTrack(old_gray, mask=facemask, **feature_params)

#Stage 3: Face detection and spatial averaging of green channel
signal = []
heartbeats = []
frame = 0
while True:
    
    ret, new_frame = video_capture.read()
    new_frame = cv2.flip(new_frame, 1)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    displayImage = new_frame.copy()

    if frame<240:
        cv2.imshow('zindagi kitni haseen hai, hena?', displayImage)
        cv2.waitKey(25)
        frame+=1
        continue

    
    # calculate optical flow 
    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params) 

    # Select good points
    good_new = new_points[st == 1]
    good_old = old_points[st == 1] 

    for i, (new, old) in enumerate(zip(good_new, good_old)): 
        cv2.circle(displayImage, (int(new.reshape(1, 2)[0][0]), int(new.reshape(1, 2)[0][1])), 5, color[i].tolist(), -1) 

    # Estimate transformation from point tracker and apply it to bounding box
    transformation = cv2.estimateAffinePartial2D(good_old, good_new)[0]
    face_points = np.array([[cx[0], cy[0]],
                           [cx[1], cy[1]],
                           [cx[2], cy[2]],
                           [cx[3], cy[3]]])
    if transformation is not None:
        transformed_points = transformation.dot(np.hstack((face_points, np.ones((4, 1)))).T).T
        cx[:] = transformed_points[:, 0]
        cy[:] = transformed_points[:, 1]

    lines=np.array([[cx[0],cy[0]], [cx[1],cy[1]], [cx[2],cy[2]], [cx[3],cy[3]]], dtype=np.int32) # Array of points of the polylines
    cv2.polylines(displayImage, [lines], True,(0,255,0), 2)

    # Create a mask with the region of interest
    roi_mask = np.zeros_like(new_frame)
    cv2.fillPoly(roi_mask, [lines], (255, 255, 255))
    roi_mask = cv2.bitwise_and(new_frame, roi_mask)
    angle_rad = math.atan2(cy[1] - cy[0], cx[1] - cx[0])
    angle_deg = math.degrees(angle_rad)
    rot_mat = cv2.getRotationMatrix2D((cx[0],cy[0]), angle_deg, 1.0)
    rx, ry = cx.copy(), cy.copy()   #rotated x, y
    r = rot_mat.dot(np.array([rx,ry, np.ones(4)]))
    rx, ry = r[0], r[1]
    roi_mask = cv2.warpAffine(roi_mask, rot_mat, dsize=(roi_mask.shape[1], roi_mask.shape[0]))
    

    greenSum = 0
    for x in range( int(rx[0]), int(rx[2]), 10):
        for y in range(int(ry[0]), int(ry[2]), 10):
            x = max(min(x, roi_mask.shape[1] - 1),0)
            y = max(min(y, roi_mask.shape[0] - 1),0)
            greenSum+=roi_mask[y][x][1]
            # Perform operations on rotated_image pixels at (x, y)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    cv2.putText(displayImage, "FPS: " + "{:.2f}".format(fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    signal.append(100*greenSum / ((int(rx[2]) - int(rx[0])) * (int(ry[2]) - int(ry[0]))))
    cv2.putText(displayImage, "Green: " + "{:.2f}".format(signal[-1]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(displayImage, "Time: " + "{:.2f}".format(frame/fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Rotated ROI Mask", roi_mask)



    cv2.imshow('zindagi kitni haseen hai, hena?', displayImage)
    k = cv2.waitKey(25) 
    if k == 27: 
        break
    if k == 32: 
        print('SUIIII')
        heartbeats.append(frame)

    frame += 1
    
    # Updating Previous frame and points 
    old_gray = new_gray.copy() 
    old_points = good_new.reshape(-1, 1, 2) 

with open('signal.txt', 'w') as f:
    for s in signal:
        f.write(str(s) + '\n')
with open('heartbeats.txt', 'w') as f:
    for h in heartbeats:
        f.write(str(h) + '\n')

video_capture.release()
cv2.destroyAllWindows()
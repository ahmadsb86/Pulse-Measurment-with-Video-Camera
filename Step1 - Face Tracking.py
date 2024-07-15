import os
import time
import cv2
import numpy as np
import math

# Init stuff
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0) 

color = np.random.randint(0, 255, (100, 3)) # For color-coding tracking points

#Stage 1: Detect Face
# cx and cy represent coordinates of the 4 corners of the face bounding box
cx = np.zeros(4)
cy = np.zeros(4)
faceFound = False
while not faceFound:
    _, old_frame = video_capture.read()
    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(old_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  
        cx = [x, x+w, x+w, x]
        cy = [y, y, y+h, y+h]
        faceFound = True


#Stage 2: Find tracking points
facemask = np.zeros_like(old_gray)
facemask = cv2.rectangle(facemask, (x, y), (x+w, y+h),255,-1)
old_points = cv2.goodFeaturesToTrack(old_gray, mask=facemask, maxCorners = 100, qualityLevel = 0.05,  minDistance = 7)

#Stage 3: Face tracking and spatial averaging of green channel
signal = [] 
markers = []    
t0 = time.time()
frameCount = 0
while True:
    
    _, new_frame = video_capture.read()
    new_frame = cv2.flip(new_frame, 1)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    displayImage = new_frame.copy()

    # 3 second delay before face tracking to adjust camera if neccessary
    if time.time() - t0 < 3:
        cv2.putText(displayImage, "Starting in 3 seconds ...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('zindagi kitni haseen hai, hena?', displayImage)
        k = cv2.waitKey(25)
        if k == 27:     #escape key
            break
        continue

    
    # calculate optical flow 
    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None) 

    # Select good points and draw them
    good_new = new_points[st == 1]
    good_old = old_points[st == 1] 

    for i, (new, old) in enumerate(zip(good_new, good_old)): 
        cv2.circle(displayImage, (int(new.reshape(1, 2)[0][0]), int(new.reshape(1, 2)[0][1])), 5, color[i].tolist(), -1) 

    # Estimate transformation from point tracker and apply it to bounding box
    transformation = cv2.estimateAffinePartial2D(good_old, good_new)[0]
    face_points = np.array([cx, cy, np.ones(4)])
    if transformation is not None:
        transformed_points = transformation.dot(face_points).T
        cx = transformed_points[:, 0]
        cy = transformed_points[:, 1]

    # Draw bounding box
    lines = np.array([(cx[i], cy[i]) for i in range(4)], dtype=np.int32)
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
    
    # Find the average green value in the ROI
    greenSum = 0
    for x in range( int(rx[0]), int(rx[2]), 10):
        for y in range(int(ry[0]), int(ry[2]), 10):
            x = max(min(x, roi_mask.shape[1] - 1),0)
            y = max(min(y, roi_mask.shape[0] - 1),0)
            greenSum+=roi_mask[y][x][1]

    # Display things on screen
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    cv2.putText(displayImage, "FPS: " + "{:.2f}".format(fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    signal.append(100*greenSum / ((int(rx[2]) - int(rx[0])) * (int(ry[2]) - int(ry[0]))))
    cv2.putText(displayImage, "Green Signal: " + "{:.2f}".format(signal[-1]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(displayImage, "Time Elapsed: " + "{:.2f}".format(time.time()-t0), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Rotated ROI Mask", roi_mask)
    cv2.imshow('zindagi kitni haseen hai, hena?', displayImage)


    k = cv2.waitKey(25) 
    if k == 27:     # escape key
        break
    if k == 32:     # space bar
        markers.append(frameCount)  #optionally used to mark important points in time 

    frameCount += 1
    
    # Updating Previous frame and points 
    old_gray = new_gray.copy() 
    old_points = good_new.reshape(-1, 1, 2) 

with open('signal.txt', 'w') as f:
    for s in signal:
        f.write(str(s) + '\n')
with open('markers.txt', 'w') as f:
    for h in markers:
        f.write(str(h) + '\n')

video_capture.release()
cv2.destroyAllWindows()
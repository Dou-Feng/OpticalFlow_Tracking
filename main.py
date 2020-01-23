import time

import cv2 as cv
import numpy as np
import opticalFlow as of

cap = cv.VideoCapture("/Users/carves/Downloads/DATA/Dancer2/img/0%03d.jpg")

ret, old_frame = cap.read()

# print(type(old_frame))

oft = of.ofTracking(old_frame)
output = old_frame
cv.imshow("tracking", output)

ix, iy, iw, ih = 135, 66, 70, 165  # simply hardcoded the values
track_window = (ix, iy, iw, ih)

rects = []

leftbtnflag = False
st = (0, 0)
tm = (0, 0)
time_start = time.time()
def drawRect(event, x, y, flag, param):
    global st, tm
    global leftbtnflag
    if event == cv.EVENT_LBUTTONDOWN:
        st = x, y
        # print("st:", st)
        leftbtnflag = True
    if event == cv.EVENT_MOUSEMOVE and leftbtnflag:
        imageCopy = output.copy()
        tm = x, y
        # print(tm)
        if tm != st:
            cv.rectangle(imageCopy, st, tm, (255, 0, 0), 2)
        cv.imshow("tracking", imageCopy)
    if event == cv.EVENT_LBUTTONUP:
        leftbtnflag = False
        print(st, tm)
        sx, sy = min(st[0], tm[0]), min(st[1], tm[1])
        sw, sh = abs(st[0] - tm[0]), abs(st[1] - tm[1])
        rects.append((sx, sy, sw, sh))
        # print("rects", rects)
        # roi = output[sy:sy+sh:, sx:sx+sw]
        # cv.imshow("ROI", roi)
        # cv.waitKey(2000)
        # cv.destroyWindow("ROI")
        time_start = time.time()


cv.setMouseCallback("tracking", drawRect)

videoPauseFlag = True
key = 0
frame_num = 0
pauseTime = 300
while 1:
    if cv.waitKey(30) & 0xFF == ord('p'):
        videoPauseFlag = ~videoPauseFlag
        time_start = time.time()
        frame_num = 0
    if videoPauseFlag:
        if pauseTime > 0:
            pauseTime = pauseTime - 1
            continue
        else:
            videoPauseFlag = False
            time_start = time.time()
            frame_num = 0
            pauseTime = 300
    if not leftbtnflag:
        ret, output = cap.read()
        frame_num = frame_num+1
    if tm != st and ret:
        rects, mask = oft.updaterects(output, rects)
        for rect in rects:
            x, y, w, h = rect
            cv.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0))
        output = cv.add(output, mask)
    if ret:
        cv.imshow("tracking", output)
    else:
        break

time_end = time.time()

print(frame_num/(time_end - time_start), "fps")

cv.destroyAllWindows()
cap.release()


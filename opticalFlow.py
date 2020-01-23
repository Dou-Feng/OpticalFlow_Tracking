import numpy as np
import cv2 as cv

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))


def inrange(x, lower, upper):
    if lower <= x <= upper:
        return x
    elif x < lower:
        return lower
    else:
        return upper

# 调用方式
# 1. 实例化ofTracking，实例化时应该传入原始帧frame(np.ndarray类型）（这一帧应该是需要offloading的一帧）
# 2. 调用updaterects函数传入当前帧frame和服务器返回的结果，得到output(矩形框列表）
class ofTracking:

    def __init__(self, frame):
        self.old_frame = frame
        self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask=None, **feature_params)

    # 更新原始帧
    # @param: 传入一帧图像
    # @output: 更新class中的一些变量的值
    def setup(self, frame):
        self.old_frame = frame
        self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask=None, **feature_params)

    # 根据运动向量更新矩形框
    # @param：frame 当前帧
    # @param: rects 矩形框list[(x, y, w, h), (x, y, w, h)...]
    # rects需要上一帧中目标所在位置的矩形框列表（由于追踪的目标可能不止一个）
    # (每个元组都有4个元素，x代表矩形框左上角的x坐标值，y代表矩形框左上角的y坐标，w和h分别代表矩形的宽度和高度）
    # @output: 返回矩形框list
    def updaterects(self, frame, rects):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **lk_params)
        # select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # 更新rects
        res = []
        weight = 1
        for rect in rects:  # rect (x, y, w, h)
            x, y, w, h = rect
            dx, dy = 0., 0.
            num = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 1)
                mask = cv.circle(mask, (a, b), 5, color[i].tolist(), -1)
                if x <= c <= x + w and y <= d <= y + h:
                    dx += a - c
                    dy += b - d
                    num = num + 1
            if num != 0:
                dx = int(dx / num * weight)
                dy = int(dy / num * weight)
            else:
                dx, dy = 0, 0
            # 考虑到矩形框必须在有效区域
            x = inrange(x+dx, 0, frame_gray.shape[0])
            y = inrange(y+dy, 0, frame_gray.shape[1])
            res.append((x, y, w, h))  # 应该要考虑矩形框不能够超出frame的图像范围

        # 更新类变量
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)

        return res, mask

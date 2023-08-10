# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:10:24 2023

@author: zafar
"""
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

# VIDEO
filename = 'test.mp4'
vidcap = cv2.VideoCapture(f'../videos/{filename}')
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
vid_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
INIT = 0
LEG_TRACK = 'R'
LEG_REF = 'L'

# GET FRAMES
list_frames = []
for i in range(vid_len):
    success,image = vidcap.read()
    if i>INIT:
        list_frames.append(image)

cv2.imshow("", list_frames[0])

#%%
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.05)
mpDraw = mp.solutions.drawing_utils
CRAD = 4
LWIDTH = 4
list_frames_out = []
list_landmarks = []
for i in range(len(list_frames)):
    img = list_frames[i].copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if i>25:
        frame_landmarks = {}
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                frame_landmarks[idx] = [cx, cy]
                # right
                if (idx in [24,26,28,30,32]):
                    cv2.circle(img, (cx, cy), CRAD, (0,0,255), -1)
                # left
                if (idx in [23,25,27,29,31]):
                    cv2.circle(img, (cx, cy), CRAD, (255,0,0), -1)
            # right
            cv2.line(img, frame_landmarks[24], frame_landmarks[26], (0,0,255), LWIDTH)
            cv2.line(img, frame_landmarks[26], frame_landmarks[28], (0,0,255), LWIDTH)
            cv2.line(img, frame_landmarks[28], frame_landmarks[30], (0,0,255), LWIDTH)
            cv2.line(img, frame_landmarks[30], frame_landmarks[32], (0,0,255), LWIDTH)
            cv2.line(img, frame_landmarks[32], frame_landmarks[28], (0,0,255), LWIDTH)
            # left
            cv2.line(img, frame_landmarks[23], frame_landmarks[25], (255,0,0), LWIDTH)
            cv2.line(img, frame_landmarks[25], frame_landmarks[27], (255,0,0), LWIDTH)
            cv2.line(img, frame_landmarks[27], frame_landmarks[29], (255,0,0), LWIDTH)
            cv2.line(img, frame_landmarks[29], frame_landmarks[31], (255,0,0), LWIDTH)
            cv2.line(img, frame_landmarks[31], frame_landmarks[27], (255,0,0), LWIDTH)
            # hip
            cv2.line(img, frame_landmarks[23], frame_landmarks[24], (255,255,255), LWIDTH)
            
            # mpDraw.draw_landmarks(img, results.pose_landmarks,
            #                            mpPose.POSE_CONNECTIONS,
            #                            mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            #                            mpDraw.DrawingSpec(color=(66,66,245), thickness=2, circle_radius=2))
            
            # Foot trace
            for f in range(len(list_landmarks)):
                if len(list_landmarks[f]) > 0:
                    cv2.circle(img, (list_landmarks[f][32][0], list_landmarks[f][32][1]), CRAD, (0,0,255), -1)
            
        cv2.imshow("Output", img)
        cv2.waitKey(1)
        
        list_landmarks.append(frame_landmarks)
        list_frames_out.append(img)

# cv2.destroyAllWindows()

#%%
print('Export Started')
video = cv2.VideoWriter(f'./export/{filename}', cv2.VideoWriter_fourcc(*'MP42'), 30.0, tuple(list_frames_out[0].shape[:2])[::-1])
for i in range(len(list_frames_out)):
    print(i)
    video.write(list_frames_out[i])

video.release()
print('Export Complete')

#%%
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import label
import pandas as pd
import seaborn as sns

xy_ax1 = np.array([list_landmarks[i][23] for i in range(len(list_landmarks))])
xy_ax2 = np.array([list_landmarks[i][24] for i in range(len(list_landmarks))])
xy_ay1 = np.array([list_landmarks[i][31] for i in range(len(list_landmarks))])
xy_ay2 = np.array([list_landmarks[i][29] for i in range(len(list_landmarks))])

xy_ax1[:,1] *= -1
xy_ax2[:,1] *= -1
xy_ay1[:,1] *= -1
xy_ay2[:,1] *= -1

vec_x = np.mean(xy_ax2 - xy_ax1, axis=0)
vec_y = np.mean(xy_ay1 - xy_ay2, axis=0)

plt.plot([0, vec_x[0]], [0, vec_x[1]])
plt.plot([0, vec_y[0]], [0, vec_y[1]])
plt.plot([vec_x[0], -1*vec_y[0] + vec_x[0]], [0, vec_y[1]])

plt.axis('equal')


#%%
# plt.plot(xy_ax1)
# plt.plot(xy_ax2)
# plt.plot(xy_ay1)
# plt.plot(xy_ay2)

pts1 = np.float32([[0, 0],
                   vec_x,
                   [-1*vec_y[0] + vec_x[0], vec_y[1]],
                   [0, vec_y[1]]])
pts1[:,0] += -1*np.mean(pts1[:,0])
pts1[:,1] += -1*np.mean(pts1[:,1])
pts2 = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])
ratio = 0.75
depth_ratio = abs(vec_y[0] / vec_y[1])
pts2[:,1] *= ratio

matrix = cv2.getPerspectiveTransform(pts1, pts2)
# result = cv2.warpPerspective(frame, matrix, (500, 600))

plt.plot(pts1[:,0], pts1[:,1])
plt.axis('equal') 

#%%
xy = np.array([list_landmarks[i][32] for i in range(len(list_landmarks))])
xy = xy - xy[0,:]
xy[:,0] *= 1
xy[:,1] *= -1

xy_o = xy.copy()
LIM = 85
pts = np.array( [ [ (xy[i, 0] - np.mean(xy[:LIM,0]),
                     xy[i, 1] - np.mean(xy[:LIM,1])) for i in range(LIM) ] ], dtype=np.float32)

xy = np.squeeze( cv2.perspectiveTransform(pts, matrix) )
vxy = np.gradient(xy)[0]

fs = 30
cf = 10
wn = cf / (fs/2)
b, a = signal.butter(2, wn)

abs_vel = np.sum(np.abs(vxy)**2,axis=-1)**(1./2)
filt_vel = signal.filtfilt(b, a, abs_vel)

THRESH_VEL = 0.03 # px/sec
THRESH_DUR = 2 # frames
lbl_vel, num_lbl = label(filt_vel[:LIM] < THRESH_VEL)

idx_vel = []
for i in range(num_lbl):
    ev_num = i+1
    ev_idx = np.where(lbl_vel==ev_num)[0]
    if len(ev_idx) >= THRESH_DUR:
        ev_vel = filt_vel[ev_idx]
        ev_idx_min = ev_idx[np.argmin(ev_vel)]
        idx_vel.append(ev_idx_min)
    else:
        print(False)

# truncate time series
idx_vel = np.array(idx_vel)
xy = xy[idx_vel[0]:idx_vel[-1]+1, :]
idx_vel = idx_vel - idx_vel[0]

# plt.plot(xy)
# plt.plot(abs_vel)

# plt.plot(filt_vel)
# plt.plot(idx_vel, filt_vel[idx_vel], 'o')
#%
plt.plot(xy[:,0], xy[:,1], 'o-')
plt.plot(xy[idx_vel,0], xy[idx_vel,1], 'o-')
plt.axis('equal')
#%%
from scipy import interpolate

interp_type = 'linear'
f_x = interpolate.interp1d(idx_vel, xy[idx_vel, 0], kind=interp_type)
f_y = interpolate.interp1d(idx_vel, xy[idx_vel, 1], kind=interp_type)
x_interp = f_x(np.linspace(0, len(xy)-1, len(xy)))
y_interp = f_y(np.linspace(0, len(xy)-1, len(xy)))

x_corr = np.array([x_interp[i] + depth_ratio*(xy[i,0] - x_interp[i]) for i in range(len(y_interp))])
y_corr = np.array([y_interp[i] + depth_ratio*(xy[i,1] - y_interp[i]) for i in range(len(y_interp))])

#%%
plt.close('all')
plt.figure()
plt.plot(xy_o[:,0], xy_o[:,1], 'o-')
plt.axis('equal')
plt.title('Raw coordinates')

plt.figure()
plt.plot(x_corr, y_corr, 'o-')
plt.plot(x_corr[idx_vel], y_corr[idx_vel], 'o')
plt.axis('equal')
plt.title('Perspective transformed coordinates')

#%% corners
from sklearn.cluster import DBSCAN

def cluster(data, epsilon,N): #DBSCAN, euclidean distance
    db     = DBSCAN(eps=epsilon, min_samples=N).fit(data)
    labels = db.labels_ #labels of the found clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
    return labels, n_clusters

corners = np.array([x_corr[idx_vel], y_corr[idx_vel]]).T
lbl_corner, n_corner = cluster(corners,epsilon=0.5,N=1)

mean_err = []
for i in range(n_corner):
    corner_pts = corners[lbl_corner==i, :]
    corner_cen = np.mean(corner_pts, axis=0)
    corner_err = corner_pts - corner_cen
    mag_err = np.sum(np.abs(corner_err)**2,axis=1)**(1./2)
    mean_err.append(np.mean(mag_err))
    
fig, axes = plt.subplots(nrows=1, ncols=2)
# corner data
axes[0].plot(x_corr, y_corr, 'k:')
for i in range(n_corner):
    axes[0].plot(corners[lbl_corner==i, 0], corners[lbl_corner==i, 1], 'o')
axes[0].axis('equal')
axes[0].set_title('Corner Detection')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')

axes[1].set_title('Corner Variability')
axes[1].set_xlabel('Corner #')
axes[1].set_ylabel('Variability')
for i in range(n_corner):
    axes[1].bar(i, mean_err[i])
    
#%%
def coordinates_to_distances(A, B, x, y):
    dy = B[1] - A[1]
    dx = B[0] - A[0]
    x_new, y_new = x * 0 + 1, y * 0 + 1
    if dx == 0:
        x_new, y_new = x * 0 + A[0], y
    if dy == 0:
        x_new, y_new = x, y * 0 + A[1]
    if dx != 0 and dy != 0:
        n = dy / dx
        m = A[1] - n * A[0]
        p = y + x / n
        x_new = (p - m) / (n + 1 / n)
        y_new = n * x_new + m
    direc = x_new * 0 - 1
    direc[(B[0] > A[0]) == (x_new > A[0])] = 1
    return direc * np.sqrt((A[0] - x_new)**2 + (A[1] - y_new)**2)

fig, axes = plt.subplots(nrows=1, ncols=2)

n_seg = len(idx_vel)-1
for i in range(n_seg):
    seg = np.array(
        [x_corr[idx_vel[i]:idx_vel[i+1]],
         y_corr[idx_vel[i]:idx_vel[i+1]]]).T
    seg_lin = np.array(
        [x_interp[idx_vel[i]:idx_vel[i+1]],
         y_interp[idx_vel[i]:idx_vel[i+1]]]).T
    A = np.array([x_interp[idx_vel[i]], y_interp[idx_vel[i]]]).T
    B = np.array([x_interp[idx_vel[i+1]], y_interp[idx_vel[i+1]]]).T
    
    L2 = np.sum((A-B)**2)
    T = np.sum((seg - A) * (B - A), axis=1) / L2
    C = np.array([A + t * (B - A) for t in T])
    
    distances = np.sum(np.abs(seg-C)**2,axis=1)**(1./2)#coordinates_to_distances(A, B, seg[:,0], seg[:,1])
    
    line, = axes[0].plot(seg[:,0], seg[:,1])
    axes[0].set_title('Segment Trajectory')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].plot([A[0], B[0]], [A[1], B[1]], color=line.get_color(), linestyle='--')
    axes[0].axis('equal')
    
    axes[1].set_title('Linear Error')
    axes[1].set_xlabel('Segment #')
    axes[1].set_ylabel('Error')
    axes[1].bar(i, np.mean(distances))
    
    # axes[1].plot(distances)
    # axes[1].plot(np.sum(np.abs(seg-seg_lin)**2,axis=1)**(1./2))
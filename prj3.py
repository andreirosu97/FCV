import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import filedialog
import os
import json 
from matplotlib import pyplot as plt

cap = cv.VideoCapture(1)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 20, (width,height))

if not cap.isOpened():
    print("Cannot open camera")
    exit()

i = 0
ret, frame = cap.read()
    # if frame is read correctly ret is True
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit(1)

previous_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
previous_frame = cv.GaussianBlur(previous_frame,(17,17),0)
i = 0
while True:
    ret, frame = cap.read()
    #skip one frame
    if (i==0):
        i = 1
        continue
    else:
        i = 0

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #getting the frame
    color_frame = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #frame alterations
    frame = cv.GaussianBlur(frame,(17,17),0) #removes noise

    #thresholding
    frame_diff = cv.absdiff(frame, previous_frame)
    ret,threshold = cv.threshold(frame_diff,25,255,cv.THRESH_BINARY)

    avg = np.average(threshold)
    if (avg <= 0):
        cv.line(color_frame,(0,0),(width, height),(0,0,255),5)
        cv.line(color_frame,(width,0),(0, height),(0,0,255),5)
    else:
        out.write(color_frame)

    previous_frame = frame.copy()
    cv.imshow('frame', color_frame)
    cv.imshow('threshold', threshold)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
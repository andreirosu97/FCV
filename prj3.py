import numpy as np
import cv2
import os
import json 
from tkinter import *
from tkinter import filedialog

root = Tk()
filename = filedialog.askopenfilename()

img = cv2.imread(filename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (152, 1))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

cntrs = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

result = img.copy()
lineCount = 1
for c in cntrs:
    box = cv2.boundingRect(c)
    x,y,w,h = box
    imageH = img.shape[0]
    if h > imageH*0.01 :
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        line = img[y:y+h, x:x+w]
        cv2.imwrite("line_" + str(lineCount) + ".png", line)
        lineCount+=1

cv2.imwrite("line_extraction_thresholding.png", thresh)
cv2.imwrite("line_extraction_dilate.png", dilate)
cv2.imwrite("line_extraction_result.png", result)

cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(1,lineCount):
    img_line = cv2.imread("line_" + str(i) + ".png")

    gray = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    erode = cv2.erode(thresh, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (1,10))
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, kernel)

    cntrs = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    result = img_line.copy()
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > 200:
            box = cv2.boundingRect(c)
            x,y,w,h = box
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        elif area > 50:
            box = cv2.boundingRect(c)
            x,y,w,h = box
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("gray", gray)
    cv2.imshow("erode", erode)
    cv2.imshow("thresh", thresh)
    cv2.imshow("dilate", dilate)
    cv2.imshow("RESULT", result)
    cv2.imwrite("line_" + str(i) + "_character_extraction_result.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
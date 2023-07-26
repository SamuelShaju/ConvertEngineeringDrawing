import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('./image/input_full.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## to remove noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# equ=cv2.equalizeHist(blur)
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

## to detect edges
edges=cv2.Canny(gray,100,200)


#horizontal line segments
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
detect_horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cntsx = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsx = cntsx[0] if len(cntsx) == 2 else cntsx[1]
original_image = cv2.imread('./image/input_full.png')
for c in cntsx:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (36,255,12), 5)


#vertical line segments
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cntsy = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsy = cntsy[0] if len(cntsy) == 2 else cntsy[1]
for c in cntsy:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (255,0,0), 5)

cv2.imshow('image',cv2.resize(original_image, (1000, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Saving the output to csv file
# import csv
# with open('cntsx.csv', 'w') as myfile:
#     for c in cntsx:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         x, y, w, h = cv2.boundingRect(c)
#         wr.writerow([x, y, x+w, y])
#         
# 
# with open('cntsy.csv', 'w') as myfile:
#     for c in cntsy:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         x, y, w, h = cv2.boundingRect(c)
#         wr.writerow([x, y, x, y+h])


# # Adding the data to common.csv file
# with open('common.csv', 'a') as myfile:
#     for c in cntsx:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         x, y, w, h = cv2.boundingRect(c)
#         wr.writerow(["Line","", -1, -1, x, y, x+w, y, -1])
# with open('common.csv', 'a') as myfile:
#     for c in cntsy:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         x, y, w, h = cv2.boundingRect(c)
#         wr.writerow(["Line","", -1, -1, x, y, x, y+h, -1])
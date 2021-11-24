import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt

# paper = cv2.imread('samples\pesepective_transform\img_4.png')
# paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)

# Coordinates that you want to Perspective Transform
pts1 = np.float32([[837,505],[818,530],[1179,563],[1152,535]])
# for i,coords in enumerate(pts1):
#     pts1[i][1] = 1080 - coords[1]

# Size of the Transformed Image
pts2 = np.float32([[800,800],[820,860],[1180,850],[1160,790]])
#for i,coords in enumerate(pts2):
#    pts2[i][1] = 1080 - coords[1]

paths = glob.glob('samples\\forza_toptruck\img_*.png')
for path in paths:

    paper = cv2.imread(path)
    paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)

    for val in pts1:
        cv2.circle(paper,(val[0],val[1]),5,(0,255,0),-1)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    print(M)
    dst = cv2.warpPerspective(paper,M,(1920,1080))
    # dst = cv2.flip(dst, 0)
    plt.imshow(dst)
    plt.show()
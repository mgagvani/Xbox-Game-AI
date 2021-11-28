import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def analyze(img):
    # the grey color we want is RGB (210,210,210) to (235,235,235)
    # mask1 = cv2.inRange(img, (200, 200, 200), (235,235,235))
    # # we also want he white pixels from 250 to 255
    # mask2 = cv2.inRange(img, (250, 250, 250), (255,255,255))
    # mask = cv2.bitwise_or(mask1, mask2)
    # target = cv2.bitwise_and(img, img, mask=mask)

    # target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Below is slow, inefficient uniform quantization
    target = img
    for i,row in enumerate(target):
        for j,pix in enumerate(row):
            # if(pix < 180):
            #     target[i][j] = 0
            # else:
            #     target[i][j] = int(pix/(255/5))*(255/5)
            if(0 <= pix <= 120): 
                target[i][j] = 0
            elif(121 <= pix <= 150):
                target[i][j] = 175
            elif(151 <= pix <= 250):
                target[i][j] = 220
            else:
                target[i][j] = 240

    print(target.shape)
    print(target.dtype)
    mask = np.zeros((302,302), dtype=np.uint8)
    target = cv2.floodFill(target, mask, (144,183), 50)[1]
    target = cv2.floodFill(target, mask, (144,166), 50)[1]
    # target[target != 50] = 0
    circle_points = []
    for i in range(0, 360, 30): # below line, y, x or i, j
        circle_points.append((142 + (30 * math.sin(math.sin(math.pi * i/180))), 179 + (30 * math.cos(math.pi * i/180))))

    for point in circle_points:
        i = int(point[0])
        j = int(point[1])
        if(target[i][j] == 220):
            target = cv2.floodFill(target, mask, (i, j), 100)[1]

    for i,row in enumerate(target):
        for j,pix in enumerate(row):
            if(pix not in [50,100]):
                target[i][j] = 255

    return target, circle_points

i = 0
while True:
    img = cv2.imread(f"samples\map_images\img_{random.randint(5,500)}.png", cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap = "gray")
    plt.show()
    img, circle_points = analyze(img)
    plt.imshow(img, cmap="gray")
    plt.scatter([point[0] for point in circle_points], [point[1] for point in circle_points])
    plt.hist(img.ravel(), 256, (0,256))
    
    plt.show()
    i+=20
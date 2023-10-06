from PIL import Image
import cv2
import numpy as np

import matplotlib.pyplot as plt
import math
from skimage.transform import hough_circle, hough_circle_peaks

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'-c tessedit_char_blacklist=~!@#$%^&*()_+|}{[]\;><.? --tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'


img = cv2.imread("samples/auctionhouse/img_21.png")

slice = np.zeros((1920,3)) # one row BGR

for x in range(1920): #loop over x
    slice[x] = img[295,x] # get 296th row


slice_g = gray = np.dot(slice[..., :3], [0.2989, 0.5870, 0.1140])

k = [-2,-1,0,1,2] # kernel
print(k)

a = math.floor(len(k)/2)

# pad array 
slice_g = np.pad(slice_g,mode="edge",pad_width=a)
# numpy.convolve exists ... 

slice_conv = np.zeros(slice_g.shape[0])

# for x in range(1,1920-1):
# for x in range(a, 1920 - a):
for x in range(0,1920-1): # since it starts at zero not one need to subtract 1 from the end
    indices = [x for x in range(x - a, x + a + 1)]

    for ki,index in enumerate(indices): # ki is index of the kernel so far
        slice_conv[x] += slice_g[index] * k[ki] 
    
    # print(slice_g[x], slice_conv[x])

# find transitions
transitions = np.empty(slice_g.shape[0])
for i,val in enumerate(slice_conv):
    if abs(val) > 100:
        transitions[i] = 1
    else:
        transitions[i] = 0

samples = [(range(slice_g.shape[0]),slice_g), (range(slice_g.shape[0]),slice_conv), (range(slice_g.shape[0]),transitions)]

plt.title("Signal 0 is original while Signal 1 is convoluted")

for i, (x, y) in enumerate(samples):
    plt.plot(x, y)
    plt.text(x[-1], y[-1], 'Signal {i}'.format(i=i)) # add annotation to plot

plt.show()

# find extents of boxes
extents = []
startx = -1
# count = 0

for i,val in enumerate(transitions):
    # print(count)
    if val == 1 and startx == -1: # first one
        startx = i
        # count += 1
    elif val == 1 and startx > -1: # not the first one
        endx = i
        
        if endx - startx > 200:
            extents.append([startx+35,endx]) # add 35 to account for offset
            # TODO change this so it throws away first two points
            startx = -1

print(extents)
# for i in extents: # find width
    # print(i[1]-i[0])

# get slices of overall image for image processsing
# first get each "box"

boxes = []
color_boxes = []

img_g = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
# img_g = np.mean(img, -1)
# img_g = cv2.imread("samples/auctionhouse/img_9.png", cv2.IMREAD_GRAYSCALE)
print(img_g.shape)

plt.imshow(img_g, cmap='gray', vmin=0, vmax=255)
plt.show()

for extent in extents:
    box = img_g[175:658, int(extent[1]-250):int(extent[1])].astype(np.uint8) 
    box_c = img[175:658, int(extent[1]-250):int(extent[1])]
    # box = cv2.threshold(box, np.mean(box), 255, cv2.THRESH_BINARY)[1]
    boxes.append(box)
    color_boxes.append(box_c)
    # print(np.mean(box))

# remove circular MT from each box
mask = cv2.imread("screenshots/mask.png")
mask  = np.dot(mask[...,:3], [0.2989, 0.5870, 0.1140])
mask = cv2.bitwise_not(mask)
print(mask.shape)

for i,box in enumerate(boxes):
    """
    x = cv2.bitwise_not(scipy.ndimage.convolve(box,mask)) * 5000
    print(x)
    plt.imshow(x, cmap='gray', vmin=0, vmax=255)
    plt.show()
    """
    """
    box_3d = np.dstack([box,box,box])
    plt.imshow(box_3d)
    plt.show()

    print(box_3d.shape)
    circles = cv2.HoughCircles(box_3d, cv2.HOUGH_GRADIENT,
                           dp=1.5, minDist=30, minRadius=15, maxRadius=60)
    print(circles)
    for x, y, r in circles[0]:
        cv2.circle(box, (x,y), r, black, 2)
    """

f, axarr = plt.subplots(1, len(boxes))
for i, img in enumerate(boxes):
    axarr[i].imshow(img, cmap='gray', vmin=0, vmax=255)

# Do OCR on each box
"""
for i,box in enumerate(boxes):
    print(i)
    print(pytesseract.image_to_string(Image.fromarray(box.astype(np.uint8)),config=tessdata_dir_config))
    print()
"""

# Show each box
plt.show()

# Further separate each box into multiple subsections so that more localized thresholding can be done
sections = [] # in same order of boxes

for i,box in enumerate(boxes):
    sections.append([]) # 2d array now
    sections[i].append(  cv2.threshold(box[5:40,63:179], np.mean(box[5:40,63:179]), 255, cv2.THRESH_BINARY)[1])         # timestamp
    sections[i].append(  cv2.threshold(box[82:117,2:122], np.mean(box[82:117,2:122]), 255, cv2.THRESH_BINARY)[1])     # winning bid
    sections[i].append(  cv2.threshold(box[82:117,123:248], np.mean(box[82:117,123:248]), 255, cv2.THRESH_BINARY)[1])   # buy it now
    sections[i].append(  cv2.threshold(box[167:200,15:53], np.mean(box[167:200,15:53]), 255, cv2.THRESH_BINARY)[1])     # player overall
    sections[i].append(  cv2.threshold(box[202:219,16:52], np.mean(box[202:219,16:52]), 255, cv2.THRESH_BINARY)[1])     # player position
    # sections[i].append(  cv2.threshold(box[425:480,11:233], np.mean(box[425:480,11:233]) - 50, 255, cv2.THRESH_BINARY)[1])   # name
    sections[i].append(  cv2.adaptiveThreshold(box[425:480,11:233], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 20))   # name
    sections[i].append(  cv2.threshold(box[168:250,185:239], np.mean(box[168:250,185:239]), 255, cv2.THRESH_BINARY)[1]) # gray collection ID image
    sections[i].append(  color_boxes[i][168:250,185:239])                                                               # color collection ID image

# get rid of MT sign in BIN and Winning Bid
for i,section in enumerate(sections):
    hough_radii = np.arange(14,16)
    hough_res = hough_circle(section[1], hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)    
    section[1] = cv2.circle(section[1], (cx[0],cy[0]), radii[0]+1, 1, -1)

    hough_res = hough_circle(section[2], hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)    
    section[2] = cv2.circle(section[2], (cx[0],cy[0]), radii[0]+1, 1, -1)

    
# erode for better OCR
for section in sections:
    section[0] = cv2.erode(section[0], np.ones((2,2), np.uint8), iterations=1)
    section[1] = cv2.erode(section[1], np.ones((3,3), np.uint8), iterations=1)
    section[2] = cv2.erode(section[2], np.ones((3,3), np.uint8), iterations=1)
    section[3] = cv2.erode(section[3], np.ones((3,3), np.uint8), iterations=1)
    section[4] = cv2.erode(section[4], np.ones((2,2), np.uint8), iterations=1)
    print(section[5][-1][-1])
    if section[5][-1][-1] > 50: # bottom right pixel is not black, so need to flip
        # section[5] = cv2.bitwise_not(section[5])
        section[5] = ~section[5]
    # section[5] = cv2.erode(section[5], np.ones((3,3), np.uint8), iterations=1)
    section[5] = cv2.morphologyEx(section[5], cv2.MORPH_CLOSE, np.ones((3,3)))
    plt.hist(section[5], [0,50,100,150,200,255])
    plt.show()
        


# show sections
for section in sections:
    f, axarr = plt.subplots(1, len(section))
    for i, img in enumerate(section):
        axarr[i].imshow(img, cmap='gray', vmin=0, vmax=255)

    # do OCR
    timestamp = pytesseract.image_to_string(Image.fromarray(section[0].astype(np.uint8)),config=tessdata_dir_config)
    winning_bid = pytesseract.image_to_string(Image.fromarray(section[1].astype(np.uint8)),config=tessdata_dir_config)
    buy_it_now = pytesseract.image_to_string(Image.fromarray(section[2].astype(np.uint8)),config=tessdata_dir_config)
    overall = pytesseract.image_to_string(Image.fromarray(section[3].astype(np.uint8)),config=tessdata_dir_config)
    position = pytesseract.image_to_string(Image.fromarray(section[4].astype(np.uint8)),config=tessdata_dir_config)
    name = pytesseract.image_to_string(Image.fromarray(section[5].astype(np.uint8)),config=tessdata_dir_config)
    
    message = f"{overall} {name} who plays {position} has {timestamp} left on auction. Winning Bid is {winning_bid} while BIN is {buy_it_now}"
    alt_message = f"""Timestamp: {timestamp}
    Winning Bid: {winning_bid}
    Buy it Now: {buy_it_now}
    Overall: {overall}
    Position: {position}
    Name: {name}"""
                      
    print(alt_message)
    
    plt.show()

    
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from openvino.inference_engine import IECore
import math

# from numba import jit

def start():
    ie = IECore()

    net = ie.read_network(model="model/road-segmentation-adas-0001.xml")
    exec_net = ie.load_network(net, "CPU")
    
    output_layer_ir = next(iter(exec_net.outputs))
    input_layer_ir = next(iter(exec_net.input_info))

    return ie, net, exec_net, output_layer_ir, input_layer_ir

# @jit(nopython = True, fastmath = True)
def find_xy(img):
    x = []
    y = []
    num_road_pixels = np.count_nonzero(img > 0) # everything but BG

    for i,row in enumerate(img[::-1]): # upside down
    # print(type(row[0]))
        if row[0].item() == 1 or row[-1].item() == 1:
            continue
        else: # road is not the first and last pixel in row
            y.append(i)

            roads = np.where(row == 1)
            # print(np.shape(roads[0][0]))
            try: 
                avg = (roads[0][0] + roads[0][-1]) / 2
                x.append(avg)
            except:
                x.append(-1)
                continue # no road here

    return x,y, num_road_pixels

def read_img_from_num(img_number):
    image = cv2.imread(f"samples/forza_badriving/img_{img_number}.png")
    return image

def inference(image, ie, net, exec_net, output_layer_ir, input_layer_ir):
    # The segmentation network expects images in BGR format
    # image = cv2.imread("data/empty_road_mapillary.jpg")
    # image = read_img_from_num(int(input("Image number: ")))
    # print(type(image))
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = image.shape
    
    # N,C,H,W = batch size, number of channels, height, width
    N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims
    
    # OpenCV resize expects the destination size as (width, height)
    resized_image = cv2.resize(image, (W, H))
    
    # reshape to network input shape
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )  
    # plt.imshow(rgb_image)
    # plt.show()
    
    # Run the infernece
    result = exec_net.infer(inputs={input_layer_ir: input_image})
    result_ir = result[output_layer_ir]
    
    # Prepare data for visualization
    segmentation_mask = np.argmax(result_ir, axis=1)
    # plt.imshow(segmentation_mask[0])
    # plt.show()
 
    img = segmentation_mask[0]

    x, y, num_road_pixels = find_xy(img)
    
    # find first index where x = -1
    for i, xval in enumerate(x):
        if xval == -1:
            last = i
            break
    
    x = x[2:last - 1] # removing first two because they are usually outliers. 
    y = y[2:last - 1] #z this can be manually tuned, for start and end

    yslope = []
    for i,item in enumerate(y):
        yslope.append(item)

    # Statistics to remove outliers
    try:
        meanx = sum(x)/len(x)
        meany = sum(y)/len(y)
        medx = x[int(len(x)/2)]
        medy = y[int(len(y)/2)]
        xs = [meanx, medx, x[0]]
        ys = [meany, medy , yslope[0]]
    except:
        return "error"

    # COMMENT ME OUT NOTE TODO when using with play.py
    # for i, item in enumerate(x):
    #     print(f"{item}\t{y[i]}")
    
    try:
        slope = (yslope[-1] - yslope[0])/(x[-1] - x[0])
    except:
        return "error"
    # print(f"First point {(x[0], y[0])}")
    # print(f"Last point {(x[-1], y[-1])}")
    # print(f"ydelta = {yslope[-1] - yslope[0]}")
    # print(f"xdelta = {x[-1] - x[0]}")

    a = math.degrees(math.atan(slope))
    if(a > 0):
        #print(f"angle is positive: {a}")
        angle = (90 - a)/a
    elif(a < 0):
        #print(f"angle is negative: {a}")
        angle = -1 * (90+a)/90
    else:
        return "error"
    
    # angle = (x[0] - x[-1])/(y[0] - y[-1]) / 5 # NOTE This parameter should be manually tuned
    #print(angle)
    
    # has to be commented out when running in play.py
    xr = list(reversed(x))
    yr = list(reversed(y))         
    plt.imshow(img[::-1])
    plt.scatter(xr,yr,c="r",s=10)
    plt.scatter(xs,ys,c="b",s=15)
    plt.gca().invert_yaxis()
    plt.show()

    print(100* num_road_pixels/(W*H)) # percent of image that is road
#   
    try:
        return angle
    except:
        return "error"

if __name__ == "__main__":
    ie, net, exec_net, output_layer_ir, input_layer_ir = start()
    while True:
        image = read_img_from_num(int(input("Image number: ")))
        t1 = time.time()
        inference(image, ie, net, exec_net, output_layer_ir, input_layer_ir)
        print(f"Inference took {time.time() - t1} seconds")

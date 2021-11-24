# ocr testing with pytesseract

import cv2
import numpy as np

import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)



def preprocess_img(img): 
    """
    Preprocesses screenshot of AH for OCR.
    Removes details which would be needed for image matching.
    """
    # grayscale
    newimg = get_grayscale(img)
    #cv2.imwrite("screenshots/img_9g.png",newimg)
    # crop
    newimg = newimg[175:660, 0:1920]
    # cv2.imwrite("screenshots/img_9gc.png",newimg)
    # increase contrast
    newimg = cv2.convertScaleAbs(newimg,alpha=1.1,beta=0) # alpha is 1 to 3
    # cv2.imwrite("screenshots/img_9gcC.png",newimg)
    # binary
    # newimg = cv2.adaptiveThreshold(newimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) 
    # cv2.imwrite("screenshots/img_9gcCb.png",newimg)

    thresh = 200
    newimg = cv2.threshold(newimg, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("screenshots/img_9gcCB_.png",newimg)
    # show img
    cv2.imshow("img",newimg)
    cv2.waitKey(0)

    return newimg

if __name__ == "__main__":
    img = cv2.imread("samples/auctionhouse/img_9.png")
    """
    cv2.imwrite("screenshots/pic1a.png",get_grayscale(img))
    cv2.imwrite("screenshots/pic1b.png",remove_noise(img))
    # cv2.imwrite("screenshots/pic1c.png",thresholding(img))
    cv2.imwrite("screenshots/pic1d.png",dilate(img))
    cv2.imwrite("screenshots/pic1e.png",erode(img))
    cv2.imwrite("screenshots/pic1f.png",opening(img))
    cv2.imwrite("screenshots/pic1g.png",canny(img))
    """

    img =  preprocess_img(img)

    print(pytesseract.image_to_string(img,config=tessdata_dir_config))
    

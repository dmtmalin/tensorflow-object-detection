import cv2
import numpy as np


class ImageProcessing:
    @staticmethod
    def __contour_is_bad(c):
        epsilon = 5.0
        approx = cv2.approxPolyDP(c, epsilon, True)
        return len(approx) <= 3

    @staticmethod
    def findContours(img, sort = True):
        im_copy = img.copy()
        gray = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if not ImageProcessing.__contour_is_bad(c)]
        if sort:
            contours = ImageProcessing.sortContours(contours)
        return contours

    @staticmethod
    def sortContours(contours, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boxes = [cv2.boundingRect(c) for c in contours]
        contours, _ = zip(*sorted(zip(contours, boxes),
                                  key=lambda b: b[1][i], reverse=reverse))
        return contours

    @staticmethod
    def boundingRectContour(c):
        poly = cv2.approxPolyDP(c, 3, True)
        return cv2.boundingRect(poly)

    @staticmethod
    def extremumContour(c):
        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bot = tuple(c[c[:, :, 1].argmax()][0])
        return left, right, top, bot


if __name__ == '__main__':
    image = cv2.imread('sample_fly.JPG')
    h, w = image.shape[:2]
    contours = ImageProcessing.findContours(image)
    blank = np.zeros((h, w, 1), dtype=np.uint8)
    mask = cv2.fillPoly(blank, contours, (255, 255, 255))
    image = cv2.bitwise_and(image, image, mask=mask)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("processing_sample_fly.JPG", image)

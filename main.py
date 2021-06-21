import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(3, frameHeight)
threshold_area = 65


def empty(a):
    pass


cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 641, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

while True:
    # img = cap.read()
    img = cv2.imread("Webp.net-resizeimage.jpg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)

    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('edit', result)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(result, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # perspective test

        docCnt = None

        pts1 = np.float32([[x, y + h], [x + w, y + h], [x, y], [x + w, y]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgoutput = cv2.warpPerspective(img, matrix, (w, h))

        # draw the biggest contour (c) in green
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    objects = 0
    text = "Obj:0"
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            objects = objects + 1
            text = "Obj:" + str(objects)

    cv2.putText(mask, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 0, 159), 1)

    # cv2.imshow('Original', img)
    cv2.imshow('HSV Color Space', img_hsv)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.imshow('Result2', imgoutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

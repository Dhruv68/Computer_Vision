import cv2
import numpy as np

min_contour_width = 48  # 40
min_contour_height = 48  # 40
offset = 3  # 10
line_height = 550  # 550
# line_height = 600  # 550
matches = []
cars = 0

c = cv2.VideoCapture('Input/traffic.mp4')
_,f = c.read()
#
avg1 = np.float32(f)
avg2 = np.float32(f)

while(1):
    _,f = c.read()
    if _ is None:
        break
    cv2.accumulateWeighted(f,avg1,0.09)
    cv2.accumulateWeighted(f,avg2,0.009)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    k = cv2.waitKey(20)

    if k == 27:
        break

cv2.destroyAllWindows()
c.release()
cv2.save("Output/Background.png", avg2)

def getGrayDiff(image,currentPoint,tmpPoint):
    return abs(int(image[currentPoint[0],currentPoint[1]]) - int(image[tmpPoint[0],tmpPoint[1]]))
#Region growth algorithm
def regional_growth (gray,seeds,threshold=5) :
    #Eight adjacent points between pixels each time the area grows
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), \
                        (0, 1), (-1, 1), (-1, 0)]
    threshold = threshold #The similarity threshold when growing, the default is that the gray level does not differ by more than 15 are considered the same
    height, weight = gray.shape
    seedMark = np.zeros(gray.shape)
    seedList = []
    for seed in seeds:
        if(seed[0] < gray.shape[0] and seed[1] < gray.shape[1] and seed[0]  > 0 and seed[1] > 0):
            seedList.append(seed)   #To be added to the list
    print(seedList)
    label = 1	#Mark the flag
    while(len(seedList)>0):     #If there are points in the list
        currentPoint = seedList.pop(0)  #Throw the first one
        seedMark[currentPoint[0],currentPoint[1]] = label   #Mark the corresponding position as 1
        for i in range(8):  # Perform similarity judgments on 8 points around this point at once
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:    #If it exceeds the limited threshold range
                continue    #Skip and continue
            grayDiff = getGrayDiff(gray,currentPoint,(tmpX,tmpY))   #Calculate the difference between the gray level of this point and the pixel point
            if grayDiff < threshold and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append((tmpX,tmpY))
    return seedMark

#Initial seed selection
def originalSeed(gray):
    ret, img1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)  # Binary graph, seed area (different divisions can get different seeds)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img1)#Perform connected domain operations, whichever is the quality point
    centroids = centroids.astype(int)#Convert to integer
    # seed = []
    # for i  in range(img1.shape[0]):
    #     for j in range(img1.shape[1]):
    #         if(img1[i,j] == 255):
    #             seed.append([i,j])

    return centroids

# # if __name__ == "__main__":
# img = cv2.imread('10.2.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# seed = originalSeed(img)
# img = regional_growth(img,seed)

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx, cy
    # return (cx, cy)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Input/Baltimore_Highway_2.mp4')
# cap = cv2.VideoCapture('Relaxing highway traffic.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('Highway_rotated.mp4',fourcc, 5, (1280,720))
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Original Video', frame)
#     # flip for truning(fliping) frames of video
#     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('Rotated video', frame)
#     cv2.imshow('Resized Rotated video', b)
#     out.write(b)
#
#     k = cv2.waitKey(30) & 0xff
#     # once you inter Esc capturing will stop
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()


cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()
cars_centroid = {}
while ret:
    d = cv2.absdiff(frame1, frame2)
    cv2.imshow("d", d)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(grey,(5,5),0)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    cv2.imshow('blur', blur)
    # ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(th, np.ones((11, 11)))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    cv2.imshow("dilated", eroded)
    # Fill any small holes
    # closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closing", closing)
    dilated_fr = cv2.dilate(eroded, np.ones((30, 30)))
    cv2.imshow("final_dilation frame", dilated_fr)

    # contours,h = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(dilated_fr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue
        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 3)

        cv2.line(frame1, (0, line_height), (1100, line_height), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)
        for (x, y) in matches:
            if y < (line_height + offset) and y > (line_height - offset):
                cars_centroid[x] = 5
                cars = cars + 1
                matches.remove((x, y))
                print(cars)

    cv2.putText(frame1, "Total vehicles Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)

    cv2.putText(frame1, "Vehicle Counting System", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 170, 0), 2)

    # cv2.drawContours(frame1,contours,-1,(0,0,255),2)

    cv2.imshow("Original", frame1)
    cv2.imshow("Threshold", th)
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret, frame2 = cap.read()
# print(matches)
cv2.destroyAllWindows()
cap.release()


#

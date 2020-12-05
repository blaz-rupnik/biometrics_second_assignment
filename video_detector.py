import cv2

cascade_ear = cv2.CascadeClassifier('cascades/haarcascade_ear_shivangbansal.xml')

# opens camera
capture = cv2.VideoCapture(0)

while 1:
    ret, img = capture.read()

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ear = cascade_ear.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

    cv2.imshow('img', img)
    # end on esc
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()

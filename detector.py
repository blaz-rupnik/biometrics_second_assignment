import cv2, sys

cascade_face = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# get image from argument
file_name = sys.argv[1]
image_name = file_name.split('/')[-1]
img = cv2.imread(file_name)

def detect_face(img):
    detection_list = cascade_face.detectMultiScale(img, 1.05, 5)
    return detection_list


def vizualization(img, detection_list):
    for x, y, w, h in detection_list:
        cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
    cv2.imwrite('outputs/detected_' + image_name, img)

# detect
detection_list = detect_face(img)
# visualize what was detected
vizualization(img, detection_list)

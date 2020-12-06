import cv2, numpy as np

# output is list of detections by given classifier
def detect(img, classifier, scale_factor, min_neighbors):
    detection_list = classifier.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return detection_list


# return numpy array in size of image, where 0 -> ear was not detected, 1 -> ear was detected
def set_detections(img, detections):
    image_detections = np.zeros((len(img), len(img[0])), dtype=bool)
    for detection in detections:
        starting_index_x = detection[0]
        starting_index_y = detection[1]
        object_width = detection[2]
        object_height = detection[3]
        for i in range(starting_index_x, starting_index_x + object_width):
            for j in range(starting_index_y, starting_index_y + object_height):
                if i < len(img) and j < len(img[0]):
                    image_detections[i][j] = True
    return image_detections

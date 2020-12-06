import cv2, sys, os, time, numpy as np
from iou_calculator import ioc_calculator
from helpers import detect, set_detections

classifier_left = cv2.CascadeClassifier("cascades/haarcascade_leftear_otsedom.xml")
classifier_right = cv2.CascadeClassifier("cascades/haarcascade_leftear_otsedom.xml")
image_paths = os.listdir("images")
number_of_inputs = float(len(image_paths))
results_file = open("results.txt", "a")
results_file.write("RESULTS FOR: otsedom (2 cascades, one for each ear) \n")

# parameters for detections
scale_factors = [1.1,1.2,1.3,1.4,1.5]
min_neighbors = [3,4,5,6]

for scale_factor in scale_factors:
    for min_neighbor in min_neighbors:      
        average_ioc = 0.0
        average_number_of_detections = 0.0
        start = time.time()
        for image_path in image_paths:
            # read image and get detections
            img = cv2.imread(f"images/{image_path}")
            detections_left = detect(img, classifier_left, scale_factor, min_neighbor)
            detections_right = detect(img, classifier_right, scale_factor, min_neighbor)
            # merge detections since we use more cascades
            detections = np.concatenate((detections_left, detections_right))
            
            # initialize img arrays for IOU calculation
            image_annotation = np.zeros((len(img), len(img[0])), dtype=bool)

            # prepare image annotation 
            annotation = cv2.imread(f"image_annot/{image_path}")
            for i in range(0, len(annotation)):
                for j in range(0, len(annotation[0])):
                    if annotation[i][j][0] > 0:
                        image_annotation[i][j] = True

            # set true in image where ear was detected
            image_detections = set_detections(img, detections)
            
            ioc = ioc_calculator(image_annotation, image_detections)
            average_ioc += ioc
            average_number_of_detections += float(len(detections))

        average_ioc = round(average_ioc / number_of_inputs, 3)
        average_number_of_detections = round(average_number_of_detections / number_of_inputs, 3)
        end = time.time()
        print(f"Finished {scale_factor}, {min_neighbor}.")
        results_file.write(f"Scale Factor: {scale_factor}, Min neighbors: {min_neighbor}, Average IOC: {average_ioc}, Average number of detections: {average_number_of_detections}, Time Elapsed: {round(end - start, 2)}\n")
    results_file.write("\n")
results_file.close()
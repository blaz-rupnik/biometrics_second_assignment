import cv2, sys, os, time, numpy as np
from iou_calculator import ioc_calculator
from helpers import detect, set_detections

classifier = cv2.CascadeClassifier(f"cascades/{sys.argv[1]}")
image_paths = os.listdir("images")
number_of_inputs = float(len(image_paths))
results_file = open("results.txt", "a")
results_file.write(f"RESULTS FOR: {sys.argv[1]} \n")

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
            detections = detect(img, classifier, scale_factor, min_neighbor)
            
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
import sys
import time

import cv2

from yolov5 import ObjectDetection

print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection('weights/yolov5n.pt', input_width=640)
print("Starting inference")
frame = cv2.imread(sys.argv[1])
frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

samples = 0
measuring_samples = 0
total_time = 0.000
while True:
    t = time.process_time()
    objs = Object_detector.detect(frame)
    frame_time = round(time.process_time() - t, 5)
    print("Frame time: ", frame_time)
    samples += 1
    if samples >= 60:
        measuring_samples += 1
        total_time += frame_time
        print("Avg Frame Time: ", round(total_time / measuring_samples, 5))
    # plotting
#    for obj in objs:
#        # print(obj)
#        label = obj['label']
#        score = obj['score']
#        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
#        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED_COLOR, 2)
#        frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                            RED_COLOR, 1, cv2.LINE_AA)
#    cv2.imshow("Result", frame)
#    cv2.waitKey(20)
import time

import numpy as np
import sys
from yolov5 import ObjectDetection

print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection('weights/' + sys.argv[1], input_width=640)
print("Starting inference")
frame = np.zeros((480, 640, 3), np.uint8)
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
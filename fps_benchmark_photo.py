import glob
import sys
import time

import cv2

use_gpio = True
try:
    import RPi.GPIO as GPIO
except ImportError:
    use_gpio = False
from yolov5 import ObjectDetection

images = []
for file in glob.glob("./Coco2017_val/*"):
    images.append(cv2.imread(file))

print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection(sys.argv[1], input_width=640)
print("Starting inference - predetect")

for i in range(0, 10):
    # Detect image few times because it is slow first few times
    objs = Object_detector.detect(images[i])

print("Starting inference")

samples = 0
total_time = 0.000
if not use_gpio:
    while True:
        for frame in images:
            t = time.time()
            objs = Object_detector.detect(frame)
            frame_time = round(time.time() - t, 5)
            # print("Frame time: ", frame_time)
            samples += 1
            total_time += frame_time
            print("Avg Frame Time: ", round(total_time / samples, 5))
            print("Average framerate:", round(samples / total_time, 2))
else:
    input_pin = 18  # BCM pin 18, BOARD pin 12
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    while True:
        while GPIO.input(input_pin) == GPIO.HIGH:
            continue
        while GPIO.input(input_pin) == GPIO.LOW:
            for frame in images:
                if GPIO.input(input_pin) == GPIO.HIGH:
                    break
                t = time.time()
                objs = Object_detector.detect(frame)
                frame_time = round(time.time() - t, 5)
                print("Frame time: ", frame_time)
                samples += 1
                total_time += frame_time
        print("Avg Frame Time: ", round(total_time / samples, 5))
        print("Total samples processed: ", samples)
        print("Average framerate:", round(samples / total_time, 2))
        samples = 0
        total_time = 0.000

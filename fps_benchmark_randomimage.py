import sys
import time

import numpy as np

use_gpio = True
try:
    import RPi.GPIO as GPIO
except ImportError:
    use_gpio = False
from yolov5 import ObjectDetection

print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection('weights/' + sys.argv[1], input_width=640)
print("Starting inference")
frame = np.random.randint(255, size=(480, 640, 3), dtype=np.uint8)

samples = 0
measuring_samples = 0
total_time = 0.000

if not use_gpio:
    while True:
        t = time.process_time()
        objs = Object_detector.detect(frame)
        frame_time = round(time.process_time() - t, 5)
        print("Frame time: ", frame_time)
        samples += 1
        measuring_samples += 1
        total_time += frame_time
        print("Avg Frame Time: ", round(total_time / measuring_samples, 5))
else:
    input_pin = 18  # BCM pin 18, BOARD pin 12
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    while True:
        while GPIO.input(input_pin) == GPIO.HIGH:
            continue
        while GPIO.input(input_pin) == GPIO.LOW:
            t = time.process_time()
            objs = Object_detector.detect(frame)
            frame_time = round(time.process_time() - t, 5)
            print("Frame time: ", frame_time)
            samples += 1
            measuring_samples += 1
            total_time += frame_time
        print("Avg Frame Time: ", round(total_time / measuring_samples, 5))
        samples = 0
        measuring_samples = 0
        total_time = 0.000

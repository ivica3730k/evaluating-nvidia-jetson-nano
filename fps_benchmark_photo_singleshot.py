import sys
import time

import cv2

use_gpio = True
try:
    import RPi.GPIO as GPIO
except ImportError:
    use_gpio = False
from yolov5 import ObjectDetection

print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection('weights/yolov5n.pt', input_width=640)
frame = cv2.imread(sys.argv[1])
frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

if not use_gpio:
    print("Inference in 1s")
    time.sleep(1)
    t = time.process_time()
    objs = Object_detector.detect(frame)
    frame_time = round(time.process_time() - t, 5)
    print("Frame time: ", frame_time)
else:
    input_pin = 18  # BCM pin 18, BOARD pin 12
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    print("Waiting for pin trigger")
    while GPIO.input(input_pin) == GPIO.LOW:
        continue
    t = time.process_time()
    objs = Object_detector.detect(frame)
    frame_time = round(time.process_time() - t, 5)
    print("Frame time: ", frame_time)

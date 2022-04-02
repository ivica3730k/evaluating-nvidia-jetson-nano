import os
import sys
import threading
import time

import cv2

sys.path.append(
    os.path.join(os.path.dirname(__file__),
                 "nvidia-jetson-power-measuring-tool"))

from jetsonpowerprofiler import jetsonpowerprofiler as powerprofiler

use_gpio = True
try:
    import RPi.GPIO as GPIO
except ImportError:
    use_gpio = False
from yolov5 import ObjectDetection

RED_COLOR = (255, 0, 0)
print("Configuring camera")
camera = cv2.VideoCapture(sys.argv[1])
ok, frame = camera.read()
if not ok:
    print("Error reading camera")
    exit(1)
print("Loading video into ram")
print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection(sys.argv[2], input_width=640)
print("Starting inference")

power_measuring_thread = threading.Thread(
    target=powerprofiler.measure_continuous, args=())

samples = 0
total_time = 0.000
if not use_gpio:
    while True:
        t = time.time()
        ok, frame = camera.read()
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
        print("Waiting for GPIO trigger")
        while GPIO.input(input_pin) == GPIO.HIGH:
            time.sleep(0.01)
            continue
        while GPIO.input(input_pin) == GPIO.LOW:
            if GPIO.input(input_pin) == GPIO.HIGH:
                break
            ok, frame = camera.read()
            if samples == 0:
                power_measuring_thread.start()
            if GPIO.input(input_pin) == GPIO.HIGH:
                break
            t = time.time()
            objs = Object_detector.detect(frame)
            frame_time = round(time.time() - t, 5)
            samples += 1
            total_time += frame_time
        powerprofiler.send_kill()
        print("Avg Frame Time: ", round(total_time / samples, 5))
        print("Total samples processed: ", samples)
        print("Average framerate:", round(samples / total_time, 2))
        print("Average power - software measured:",
              powerprofiler.get_average_power())
        powerprofiler.clean()
        del power_measuring_thread
        power_measuring_thread = threading.Thread(
            target=powerprofiler.measure_continuous, args=())
        samples = 0
        total_time = 0.000

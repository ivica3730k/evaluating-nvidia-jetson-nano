import glob
import multiprocessing
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "nvidia-jetson-power-measuring-tool"))
import cv2
from jetsonpowerprofiler import jetsonpowerprofiler as powerprofiler

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

power_measuring_thread = multiprocessing.Process(target=powerprofiler.measure_continuous, args=())

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
        print("Waiting for GPIO trigger")
        while GPIO.input(input_pin) == GPIO.HIGH:
            time.sleep(0.001)
            continue
        while GPIO.input(input_pin) == GPIO.LOW:
            for frame in images:
                if power_measuring_thread.is_alive() is False:
                    power_measuring_thread.start()
                if GPIO.input(input_pin) == GPIO.HIGH:
                    break
                t = time.time()
                objs = Object_detector.detect(frame)
                frame_time = round(time.time() - t, 5)
                print("Frame time: ", frame_time)
                samples += 1
                total_time += frame_time
        power_measuring_thread.terminate()
        print("Avg Frame Time: ", round(total_time / samples, 5))
        print("Total samples processed: ", samples)
        print("Average framerate:", round(samples / total_time, 2))
        print("Average power - software measured:", powerprofiler.get_average_power())
        powerprofiler.clean()
        samples = 0
        total_time = 0.000

import glob
import os
import sys
import threading
import time

import cv2

sys.path.append(
    os.path.join(os.path.dirname(__file__),
                 "NvidiaJetsonNanoPowerMeasuringTool"))
from jetsonpowerprofiler import jetsonpowerprofiler as powerprofiler

sys.path.append(os.path.join(os.path.dirname(__file__), "AzureVisionTools"))
from AzureVisionTools import AzureObjectDetectionEngine as AzureObjectDetection

# free tier
ENDPOINT = "https://n0781349-cv.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = "10e1d0d7661e495b8f2bc696f74a34fa"
# Paid tier
#ENDPOINT = "https://n0781349-cv-paid.cognitiveservices.azure.com/"
#SUBSCRIPTION_KEY = "665384b591b348528c950d1a6fa0336e"
AzureObjectDetection.load_credentials(ENDPOINT, SUBSCRIPTION_KEY)

_use_gpio = True
try:
    import RPi.GPIO as GPIO
except ImportError:
    _use_gpio = False

images = []
for file in glob.glob("./Coco2017_val/*"):
    images.append(cv2.imread(file))

print("Starting inference")

_power_measuring_thread = threading.Thread(
    target=powerprofiler.measure_continuous, args=())

samples = 0
total_time = 0.000
if not _use_gpio:
    while True:
        for frame in images:
            t = time.time()
            try:
                objs = AzureObjectDetection.inference_from_cv2_image(frame)
            except:
                break
            frame_time = round(time.time() - t, 5)
            print("Frame time: ", frame_time)
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
                if samples == 0:
                    _power_measuring_thread.start()
                if GPIO.input(input_pin) == GPIO.HIGH:
                    break
                t = time.time()
                try:
                    objs = AzureObjectDetection.inference_from_cv2_image(frame)
                except:
                    break
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
        del _power_measuring_thread
        _power_measuring_thread = threading.Thread(
            target=powerprofiler.measure_continuous, args=())
        samples = 0
        total_time = 0.000

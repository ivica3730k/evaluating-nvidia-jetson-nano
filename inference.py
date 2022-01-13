import cv2

from yolov5 import ObjectDetection

RED_COLOR = (255, 0, 0)
print("Configuring camera")
camera = cv2.VideoCapture(0)
ok, frame = camera.read()
if not ok:
    print("Error reading camera")
    exit(1)
print("Loading weigths!")
Object_detector = ObjectDetection.ObjectDetection('weights/yolov5s.pt', input_width=640)
print("Starting inference")

while True:
    print("Inference starting:")
    ok, frame = camera.read()
    if not ok:
        print("Error reading camera")
        exit(1)
    objs = Object_detector.detect(frame)

    # plotting
    for obj in objs:
        print(obj)
        # print(obj)
        label = obj['label']
        score = obj['score']
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED_COLOR, 2)
        frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            RED_COLOR, 1, cv2.LINE_AA)
    cv2.imshow("Result", frame)
    cv2.waitKey(20)

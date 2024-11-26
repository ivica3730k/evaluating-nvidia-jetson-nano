import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utilities.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObjectDetection:

    def __init__(self,
                 model_path,
                 input_width=320,
                 conf_threshold=0.25,
                 iou_thres=0.45):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        self.input_width = input_width
        self.conf_threshold = conf_threshold
        self.iou_thres = iou_thres

    def detect(self, main_img):
        height, width = main_img.shape[:2]
        new_height = int((((self.input_width / width) * height) // 32) * 32)

        img = cv2.resize(main_img, (self.input_width, new_height))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_threshold,
                                   iou_thres=self.iou_thres,
                                   classes=None)
        items = []

        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(), 2)
                # label = self.classes[int(p[5])]
                label = int(p[5])
                xmin = int(p[0] * main_img.shape[1] / self.input_width)
                ymin = int(p[1] * main_img.shape[0] / new_height)
                xmax = int(p[2] * main_img.shape[1] / self.input_width)
                ymax = int(p[3] * main_img.shape[0] / new_height)

                item = {
                    'label': label,
                    'bbox': [(xmin, ymin), (xmax, ymax)],
                    'score': score
                }

                items.append(item)

        return items


if __name__ == "__main__":
    RED_COLOR = (255, 0, 0)
    print("Configuring camera")
    camera = cv2.VideoCapture(0)
    ok, frame = camera.read()
    if not ok:
        print("Error reading camera")
        exit(1)
    print("Loading weigths!")
    Object_detector = ObjectDetection(str(Path(__file__).parent.absolute()) +
                                      '/weights/yolov5m.pt',
                                      input_width=640)
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
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED_COLOR,
                                  2)
            frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin, ymin),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED_COLOR, 1,
                                cv2.LINE_AA)
        cv2.imshow("Result", frame)
        cv2.waitKey(20)

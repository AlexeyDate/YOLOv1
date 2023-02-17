import sys
import getopt
import time

import cv2
import torch
import albumentations as A
import albumentations.pytorch
import random

from model.model import YOLOv1
from tools.utils import non_max_supression
from tools.utils import convert_to_yolo

data_test = './310.jpg'
data_label = './data/classes.names'
iou_threshold = 0.5
threshold = 0.15
width = 448
height = 448
classes = 4
yolo_weights_path = './weights/yolov1.pt'
show = False
image = True
out = None

config_list = ['data_test=',
               'data_label=',
               'classes=',
               'yolo_weights_path=',
               'show',
               'video',
               'output=']

try:
    options, args = getopt.getopt(sys.argv[1:], '', config_list)
    for opt, arg in options:
        if opt in ['--data_test']:
            data_test = arg
        if opt in ['--data_label']:
            data_label = arg
        if opt in ['--classes']:
            classes = int(arg)
        if opt in ['--yolo_weights_path']:
            yolo_weights_path = arg
        if opt in ['--show']:
            show = True
        if opt in ['--video']:
            image = False
        if opt in ['--output']:
            out = arg
except getopt.GetoptError:
    print("configuration error, please check your flags and values")
    sys.exit(-1)

transform = A.Compose(
     [
        A.Resize(width, height),
        A.Normalize(),
        A.pytorch.ToTensorV2()
     ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name())

model = YOLOv1(S=7, B=2, C=classes, device=device).to(device)

if yolo_weights_path is not None:
    model.load_state_dict(torch.load(yolo_weights_path))


cap = cv2.VideoCapture(data_test)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if out and not image:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out, fourcc, 25.0, (width, height))

frame_counter = 0
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        transformed_image = transformed_image.unsqueeze(dim=0)
        transformed_image = transformed_image.to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(transformed_image)
        pred_boxes = []
        S = predictions.size(1)
        for y in range(0, S):
            for x in range(0, S):
                pred_box = torch.empty((5 + classes), dtype=torch.float32)
                pred_box[:5] = predictions[0, y, x, :5]
                pred_box[5:] = predictions[0, y, x, 10:]
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                pred_boxes.append(pred_box)

                pred_box = torch.empty((5 + classes), dtype=torch.float32)
                pred_box = predictions[0, y, x, 5:]
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                pred_boxes.append(pred_box)

        pred_boxes = non_max_supression(pred_boxes, iou_threshold, threshold)

        labels = [[str, tuple] for i in range(classes)]
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (100, 255, 40)]
        with open(data_label, 'r') as f:
            for line in f:
                (val, key) = line.split()
                labels[int(val)][0] = key

                if int(val) < len(colors):
                    labels[int(val)][1] = colors[int(val)]
                else:
                    labels[int(val)][1] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        height, width, _ = frame.shape
        for box in pred_boxes:
            conf = box[0].item()
            x1 = int(box[1] * width - box[3] * width / 2)
            y1 = int(box[2] * height - box[4] * height / 2)
            x2 = int(box[1] * width + box[3] * width / 2)
            y2 = int(box[2] * height + box[4] * height / 2)
            choose_class = torch.argmax(box[5:])

            line_thickness = 2
            text = labels[choose_class][0] + ' ' + str(round(conf, 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=labels[choose_class][1], thickness=line_thickness)
            size, baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, fontScale=0.5, thickness=1)
            text_w, text_h = size
            cv2.rectangle(frame, (x1, y1), (x1 + text_w + line_thickness, y1 + text_h + baseline),
                          color=labels[choose_class][1], thickness=-1)
            cv2.putText(frame, text, (x1 + line_thickness, y1 + 2 * baseline + line_thickness), cv2.FONT_ITALIC,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=9)

        if show:
            cv2.imshow('Detect', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if out:
            if image:
                cv2.imwrite(out, frame)
            else:
                out_video.write(frame)
        if not image:
            frame_counter += 1
            current_time = time.time() - start
            if current_time >= 1:
                print("FPS:", frame_counter)
                start = time.time()
                frame_counter = 0
    else:
        break

if out is not None and not image:
    out_video.release()

if show:
    if image:
        cv2.waitKey(0)
    cap.release()
    cv2.destroyWindow('Detect')

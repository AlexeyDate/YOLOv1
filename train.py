import sys
import getopt

import albumentations as A
import albumentations.pytorch
import torch
from torch.utils.data import DataLoader

from tools.utils import get_bound_boxes
from tools.mAP import mean_average_precision
from tools.fit import fit
from model.dataset import Dataset
from model.model import YOLOv1
from model.loss import YoloLoss


data_train = './data/train'
data_test = './data/test'
data_label = './data/classes.names'
extraction_weights_path = None
yolo_weights_path = None
file_format = 'txt'
width = 448
height = 448
batch_size = 8
epochs = 100
classes = 4
learning_rate = 5e-5
weight_decay = 0.0005
convert_to_yolo = False
save = False

config_list = ['data_train=',
               'data_test=',
               'data_label=',
               'extraction_weights_path=',
               'yolo_weights_path=',
               'file_format=',
               'width=',
               'height=',
               'batch_size=',
               'epochs=',
               'classes=',
               'learning_rate=',
               'weight_decay=',
               'convert_to_yolo',
               'save']
try:
    options, args = getopt.getopt(sys.argv[1:], '', config_list)
    for opt, arg in options:
        if opt in ['--data_train']:
            data_train = arg
        if opt in ['--data_test']:
            data_test = arg
        if opt in ['--data_label']:
            data_label = arg
        if opt in ['--extraction_weights_path']:
            extraction_weights_path = arg
        if opt in ['--yolo_weights_path']:
            yolo_weights_path = arg
        if opt in ['--file_format']:
            file_format = arg
        if opt in ['--width']:
            width = int(arg)
        if opt in ['--height']:
            height = int(arg)
        if opt in ['--batch_size']:
            batch_size = int(arg)
        if opt in ['--epochs']:
            epochs = int(arg)
        if opt in ['--classes']:
            classes = int(arg)
        if opt in ['--learning_rate']:
            learning_rate = float(arg)
        if opt in ['--weight_decay']:
            weight_decay = float(arg)
        if opt in ['--convert_to_yolo']:
            convert_to_yolo = True
        if opt in ['--save']:
            save = True
except getopt.GetoptError:
    print("configuration error, please check your flags and values")
    sys.exit(-1)

train_transform = A.Compose(
    [
        A.Resize(width, height),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

test_transform = A.Compose(
    [
        A.Resize(width, height),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_dataset = Dataset(
    transforms=train_transform,
    data_dir=data_train,
    labels_dir=data_label,
    S=7,
    C=classes,
    file_format=file_format,
    convert_to_yolo=convert_to_yolo
)

val_dataset = Dataset(
    transforms=test_transform,
    data_dir=data_test,
    labels_dir=data_label,
    S=7,
    C=classes,
    file_format=file_format,
    convert_to_yolo=convert_to_yolo
)

# a few checks to make sure the solution is correct
assert isinstance(train_dataset[0], dict)
assert len(train_dataset[0]) == 2
assert isinstance(train_dataset[0]['image'], torch.Tensor)
assert isinstance(train_dataset[0]['target'], torch.Tensor)
print('all tests is correct')

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name())

if extraction_weights_path is not None:
    model = YOLOv1(S=7, B=2, C=classes, device=device, extraction_weights_path=extraction_weights_path).to(device)
else:
    model = YOLOv1(S=7, B=2, C=classes, device=device).to(device)

if yolo_weights_path is not None:
    model.load_state_dict(torch.load(yolo_weights_path))

Yolo_loss = YoloLoss(S=7, B=2, C=classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

fit(model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=Yolo_loss,
    epochs=epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device)

pred_boxes, true_boxes = get_bound_boxes(train_dataloader, model, iou_threshold=0.5, threshold=0.15, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'train mAP: {mAP}\n')

pred_boxes, true_boxes = get_bound_boxes(val_dataloader, model, iou_threshold=0.5, threshold=0.15, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'validation mAP: {mAP}\n')

if save:
    torch.save(model.state_dict(), './yolov1_' + str(epochs) + '.pt')

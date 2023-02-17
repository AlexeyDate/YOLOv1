import torch
from tqdm import tqdm


def intersection_over_union(predicted_bbox, ground_truth_bbox) -> float:
    """
    Intersection Over Union for 2 rectangles

    param: predicted_bbox - [x, y, w, h]
    param: ground_truth_bbox - [x, y, w, h]

    return: Intersection Over Union
    """

    # convert values to x1, y1, x2, y2
    predicted_bbox_x1 = predicted_bbox[0] - predicted_bbox[2] / 2
    predicted_bbox_y1 = predicted_bbox[1] - predicted_bbox[3] / 2
    predicted_bbox_x2 = predicted_bbox[0] + predicted_bbox[2] / 2
    predicted_bbox_y2 = predicted_bbox[1] + predicted_bbox[3] / 2
    ground_truth_bbox_x1 = ground_truth_bbox[0] - ground_truth_bbox[2] / 2
    ground_truth_bbox_y1 = ground_truth_bbox[1] - ground_truth_bbox[3] / 2
    ground_truth_bbox_x2 = ground_truth_bbox[0] + ground_truth_bbox[2] / 2
    ground_truth_bbox_y2 = ground_truth_bbox[1] + ground_truth_bbox[3] / 2

    intersection_bbox = torch.tensor(
        [
            max(predicted_bbox_x1, ground_truth_bbox_x1),
            max(predicted_bbox_y1, ground_truth_bbox_y1),
            min(predicted_bbox_x2, ground_truth_bbox_x2),
            min(predicted_bbox_y2, ground_truth_bbox_y2),
        ]
    )

    intersection_area = max(intersection_bbox[2] - intersection_bbox[0], 0) * max(
        intersection_bbox[3] - intersection_bbox[1], 0
    )
    area_predicted = (predicted_bbox_x2 - predicted_bbox_x1) * (predicted_bbox_y2 - predicted_bbox_y1)
    area_gt = (ground_truth_bbox_x2 - ground_truth_bbox_x1) * (ground_truth_bbox_y2 - ground_truth_bbox_y1)

    union_area = area_predicted + area_gt - intersection_area

    iou = intersection_area / union_area
    return iou


def non_max_supression(bboxes, iou_threshold, threshold):
    bboxes = [box for box in bboxes if box[0] >= threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    non_max_bboxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        non_max_bboxes.append(current_box)

        temp_bboxes = []
        for box in bboxes:
            class_box = torch.argmax(box[5:])
            class_current_box = torch.argmax(current_box[5:])

            if intersection_over_union(current_box[1:5], box[1:5]) < iou_threshold or class_box != class_current_box:
                temp_bboxes.append(box)

        bboxes = temp_bboxes

    return non_max_bboxes


def convert_to_yolo(bbox, grid_cell_x, grid_cell_y, S):
    """
    convert prediction coordinates to yolo standart format
    x, y relative to grid cell convert to relative to image
    """

    bbox[0] = (bbox[0] + grid_cell_x) / S
    bbox[1] = (bbox[1] + grid_cell_y) / S
    bbox[2]
    bbox[3]
    return bbox


def get_bound_boxes(loader, model, iou_threshold=0.5, threshold=0.4, device='cpu'):
    """
    get prediction and target bound boxes with non-maximum supression

    param: loader - dataloader
    param: model - model
    param: device - device of the initialized model (cpu or gpu)
    param: iou_threshold - threshold of IOU (default = 0.5)
    param: threshold - threshold of confidience (default = 0.4)

    return: all prediction bound boxes, all true bound boxes
    """

    assert isinstance(loader, torch.utils.data.dataloader.DataLoader),\
        "loader does not match the type of torch.utils.data.dataloader.DataLoader"

    model.eval()
    for i, batch in enumerate(tqdm(loader, desc=f'Prediction all bound boxes', leave=False)):
        images = batch['image'].to(device)
        if i == 0:
            targets = batch['target'].to(device)
            with torch.no_grad():
                predictions = model(images)
        else:
            target = batch['target'].to(device)
            targets = torch.cat((targets, target))
            with torch.no_grad():
                predictions = torch.cat((predictions, model(images)))

    size = predictions.size(0)
    S = predictions.size(1)
    all_pred_boxes = []
    all_true_boxes = []
    for i in range(0, size):
        image_pred_boxes = []
        image_true_boxes = []
        for y in range(0, S):
            for x in range(0, S):
                if targets[i, y, x, 0] == 1:
                    pred_box = targets[i, y, x]
                    pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                    image_true_boxes.append(pred_box)

                pred_box = torch.empty(targets.size(-1), dtype=torch.float32)
                pred_box[:5] = predictions[i, y, x, :5]
                pred_box[5:] = predictions[i, y, x, 10:]
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                image_pred_boxes.append(pred_box)

                pred_box = torch.empty(targets.size(-1), dtype=torch.float32)
                pred_box = predictions[i, y, x, 5:]
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                image_pred_boxes.append(pred_box)

        image_pred_boxes = non_max_supression(image_pred_boxes, iou_threshold, threshold)
        all_pred_boxes.append(image_pred_boxes)
        all_true_boxes.append(image_true_boxes)

    return all_pred_boxes, all_true_boxes




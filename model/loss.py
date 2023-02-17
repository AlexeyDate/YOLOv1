import torch
from torch import nn
from tools.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    """"
    target = [P, x, y, w, h, c1, c2, c3] 
    prediction = [P, x, y, w, h, P, x, y, w, h, c1, c2, c3]
    expected shape of predictions = S * S * (5 * B + C) * batch_size
    """

    def forward(self, predictions, target):

        batch_size = target.size(0)
        loss_xy = 0
        loss_wh = 0
        loss_obj = 0
        loss_no_obj = 0
        loss_class = 0

        for i in range(batch_size):
            for y in range(self.S):
                for x in range(self.S):
                    if target[i, y, x, 0] == 1:
                        target_box = target[i, y, x]
                        pred_box = predictions[i, y, x]
                        iou1 = intersection_over_union(pred_box[1:5], target_box[1:5])
                        iou2 = intersection_over_union(pred_box[6:10], target_box[1:5])

                        if iou1 > iou2:
                            iou = iou1
                            selected_box = pred_box[:5]
                            selected_box = torch.cat((selected_box, pred_box[10:]))
                            unselected_box = pred_box[5:]
                        else:
                            iou = iou2
                            selected_box = pred_box[5:]
                            unselected_box = pred_box[:5]
                            unselected_box = torch.cat((unselected_box, pred_box[10:]))

                        loss_xy += self.mse(selected_box[1:3], target_box[1:3])
                        loss_wh += self.mse(torch.sign(selected_box[3:5]) * torch.sqrt(torch.abs(selected_box[3:5])),
                                            target_box[3:5].sqrt())

                        loss_obj += (selected_box[0] - iou * 1) ** 2

                        loss_no_obj += (unselected_box[0] - 0) ** 2

                        loss_class += self.mse(selected_box[5:], target_box[5:])

                    else:
                        loss_no_obj += torch.sum((predictions[i, y, x, [0, 5]] - 0) ** 2)

        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_no_obj + loss_class
        return loss / batch_size

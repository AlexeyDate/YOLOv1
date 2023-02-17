import os
import xmltodict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List


class Dataset(Dataset):
    def __init__(self, data_dir, labels_dir, transforms, S=7, C=3, file_format='txt', convert_to_yolo=True):
        self.class2tag = {}
        with open(labels_dir, 'r') as f:
            for line in f:
                (val, key) = line.split()
                self.class2tag[key] = val

        self.image_paths = []
        self.box_paths = []
        for tag in self.class2tag:
            for file in os.listdir(data_dir + '/' + tag):
                if file.endswith('.jpg'):
                    self.image_paths.append(data_dir + '/' + tag + '/' + file)
                if file.endswith('.' + file_format):
                    self.box_paths.append(data_dir + '/' + tag + '/' + file)

        # sorting to access values by equivalent files
        self.image_paths = sorted(self.image_paths)
        self.box_paths = sorted(self.box_paths)

        assert len(self.image_paths) == len(self.box_paths)

        self.transforms = transforms
        self.S = S
        self.C = C
        self.file_format = file_format
        self.convert_to_yolo = convert_to_yolo

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        if self.file_format == 'xml':
            bboxes, class_labels = self.__get_boxes_from_xml(self.box_paths[idx])
        if self.file_format == 'txt':
            bboxes, class_labels = self.__get_boxes_from_txt(self.box_paths[idx])

        if self.convert_to_yolo:
            for i, box in enumerate(bboxes):
                bboxes[i] = self.__convert_to_yolo_box_params(box, image.shape[1], image.shape[0])

        transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = torch.tensor(transformed['bboxes'])
        transformed_class_labels = torch.tensor(transformed['class_labels'])

        """
        create a target matrix 
        each grid cell = [P, x, y, w, h, c1, c2, c3]
        size of grid cell = S * S
        if we have more then one box in grid cell then we choose the last box

        x, y values are calculated relative to the grid cell
        """
        target = torch.tensor([[0] * (5 + self.C)] * self.S * self.S, dtype=torch.float32)
        target = target.reshape((self.S, self.S, (5 + self.C)))

        for i, box in enumerate(transformed_bboxes):
            class_tensor = torch.zeros(self.C, dtype=torch.float32)
            class_tensor[transformed_class_labels[i]] = 1
            x_cell = int(self.S * box[0])
            y_cell = int(self.S * box[1])
            target[y_cell, x_cell] = torch.cat((torch.tensor(
                [
                    1,
                    self.S * box[0] - x_cell,
                    self.S * box[1] - y_cell,
                    box[2],
                    box[3]
                ]
            ), class_tensor), dim=0)

        return {"image": transformed_image, "target": target}

    def __len__(self):
        return len(self.image_paths)

    def __get_boxes_from_txt(self, txt_filename: str):
        boxes = []
        class_labels = []

        with open(txt_filename) as f:
            for obj in f:
                param_list = list(map(float, obj.split()))

                boxes.append(param_list[1:])
                class_labels.append(int(param_list[0]))

        return boxes, class_labels

    def __get_boxes_from_xml(self, xml_filename: str):
        boxes = []
        class_labels = []

        with open(xml_filename) as f:
            xml_content = xmltodict.parse(f.read())
        xml_object = xml_content['annotation']['object']

        if type(xml_object) is dict:
            xml_object = [xml_object]

        if type(xml_object) is list:
            for obj in xml_object:
                boxe_list = list(map(float, [obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'],
                                             obj['bndbox']['ymax']]))
                boxes.append(boxe_list)
                class_labels.append(self.class2tag[obj['name']])

        return boxes, class_labels

    def __convert_to_yolo_box_params(self, box_coordinates: List[int], im_w, im_h):
        ans = list()

        ans.append((box_coordinates[0] + box_coordinates[2]) / 2 / im_w)  # x_center
        ans.append((box_coordinates[1] + box_coordinates[3]) / 2 / im_h)  # y_center

        ans.append((box_coordinates[2] - box_coordinates[0]) / im_w)  # width
        ans.append((box_coordinates[3] - box_coordinates[1]) / im_h)  # height
        return ans

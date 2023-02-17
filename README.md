# Implementation YOLOv1 using Pytorch
![image1](https://user-images.githubusercontent.com/86290623/219783934-c0553408-48f6-474c-bd45-05b65c450e5b.jpg)

## Dataset
* This repository was train on the [African Wildlife Dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) from Kaggle
* The data folder must contain the train and test folders as follows:
> 
    ├── data 
      ├── train
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
      ├── test 
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
        
## Training
> 
    python3 train.py --data_train data/train --data_test data/test --data_label data/classes.names --epochs 100 --batch_size 32 --learning_rate 1e-5 --weight_decay 0.0005 --classes 4 --save

Other train parameters:
* --yolo_weights_path            (states: path to yolo pytorch weights)
* --yolo_extraction_weights_path (states: path to extraction binary weights, it's base CNN module of yolo)
* --format_file                  (states: txt or xml)
* --convert_to_yolo              (states: exist this flag or no) if you need convert bounding boxes in the dataset to yolo format
* --width                        (states: width of image on 1 layer)
* --height                       (states: height of image on 1 layer)

## Inference
On video:
> 
    python3 detect.py --video --data_test content/video.mp4 --output content/detect.mp4 --data_label data/classes.names --yolo_weights_path weights/yolov1.pt --show
On image:
> 
    python3 detect.py --data_test content/image.jpg --output content/detect.jpg --data_label data/classes.names --yolo_weights_path weights/yolov1.pt --show

![image2](https://user-images.githubusercontent.com/86290623/219795511-21dd8c57-387d-45cf-8f28-6bcee6621fee.jpg)

## Specificity
* Using batch normalization in model architecture

## References
* [Extraction weights from ImageNet](https://pjreddie.com/media/files/extraction.weights)
___
* [African Wildlife dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife?resource=download)
* [African Wildlife Pytorch weights](https://drive.google.com/file/d/1V8tXqC5kN1WM3I7kk-w56RNFyL9oDAQV/view?usp=share_link)
* [Optimizer African Wildlife state](https://drive.google.com/u/1/uc?id=1V8tXqC5kN1WM3I7kk-w56RNFyL9oDAQV&export=download)

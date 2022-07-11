# Confused class for YOLOv5
This is a Pytorch implementation of **Confused class cheking module** for [YOLOv5](https://github.com/ultralytics/yolov5).

Can't see all of them to understand data features. It can be numerically checking how similar it is to which class and how much it is predicted with confidence score-based custom metric for model improvement.

- Calculate the variance of the confidence score within each image
- The smaller the variance value, the more confusing the class

![confused1](https://user-images.githubusercontent.com/87693860/178217459-85d88a34-2dc9-4f6c-943e-170e84e530ba.PNG)

<br>

![confused2](https://user-images.githubusercontent.com/87693860/178217467-b8227dd7-7eea-4f88-86d8-437fcf646ec7.PNG)

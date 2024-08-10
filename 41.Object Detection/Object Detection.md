## 目标检测方法

### 传统目标检测方法
- Viola Jones检测器
- HOG检测器
- DPM检测器

### 基于深度学习的目标检测方法
#### 两阶段算法
- R-CNN
- SPP-Net
- Fast R-CNN
- Faster R-CNN
- FPN
#### 一阶段算法
- YOLO
- SSD
- RetinaNet
- FCOS


## 目标检测基本概念
### IoU
- IoU（Intersection over Union）是目标检测中常用的评价指标，用于衡量预测框与真实框的重叠程度。IoU的值介于0和1之间，值越大表示预测框与真实框的重叠程度越高。

## NMS操作
NMS（Non-Maximum Suppression）是一种常用的目标检测后处理技术，用于去除重叠的预测框。NMS的步骤如下：
1. 对所有预测框按照置信度进行排序，置信度最高的预测框排在最前面。
2. 选择置信度最高的预测框，将其标记为保留，并将其从预测框列表中删除。
3. 计算保留的预测框与剩余预测框的IoU，如果IoU大于设定的阈值，则将对应的预测框从预测框列表中删除。

## 目标检测评价指标
### TP、FP、TN、FN
- TP（True Positive）：预测为正样本，且预测正确。
- FP（False Positive）：预测为正样本，但预测错误。
- TN（True Negative）：预测为负样本，且预测正确。
- FN（False Negative）：预测为负样本，但预测错误。
### Precision
- Precision（精确率）是预测为正样本中，预测正确的比例。计算公式为：
$$
Precision = \frac{TP}{TP + FP}
$$
### Recall
- Recall（召回率）是真实为正样本中，预测正确的比例。计算公式为：
$$
Recall = \frac{TP}{TP + FN}
$$

### AP
- PR曲线下的面积
- PR曲线为置信度从0.5变化到1情况下的**Precision-Recall曲线**，横轴为Recall，纵轴为Precision。
### mAP
- 各类别的AP取平均


## 目标检测常用数据集
- PASCAL VOC
- COCO
- ImageNet



#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
for i in range(len(videoFiles)):
    cam = cv2.VideoCapture(videoFile)
    currentFrame = 0
    while(True):
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(currentFrame + '.jpg', frame)
            currentFrame += 1
        else:
            break
    cam.release()


# In[ ]:


# image resizing 

from PIL import Image
for image_file in images:
    image = Image.open(image_file)
    resize_image = image.resize((192, 108))
    resize_image.save(new_path)

# label resizing

def changeLabel(xmlPath, newXmlPath, imgPath, boxes):
    tree = elemTree.parse(xmlPath)

    # path 변경
    path = tree.find('./path')
    path.text = imgPath[0]

    # bounding box 변경
    objects = tree.findall('./object')
    for i, object_ in enumerate(objects):
        bndbox = object_.find('./bndbox')
        bndbox.find('./xmin').text = str(boxes[i][0])
        bndbox.find('./ymin').text = str(boxes[i][1])
        bndbox.find('./xmax').text = str(boxes[i][2])
        bndbox.find('./ymax').text = str(boxes[i][3])
    tree.write(newXmlPath, encoding='utf8')


# In[ ]:


#horizontal filp 

import random
import numpy as np
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = img_center.astype(int)
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
        return img, bboxes
# label,image gathering 

for imgFile in imgFiles:
    fileName = imgFile.split('.')[0]
    label = f'{labelPath}{fileName}.xml'
    w, h = getSizeFromXML(label)

    # opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
    image = cv2.imread(imgPath + imgFile)[:,:,::-1]
    bboxes = getRectFromXML(classes, label)

    # HorizontalFlip image
    image, bboxes = RandomHorizontalFlip(1)(image.copy(), bboxes.copy())

    # Save image
    image = Image.fromarray(image, 'RGB')
    newImgPath = f'./data/light/image/train/{className}/'
    if not os.path.exists(newImgPath):
        os.makedirs(newImgPath)
    image.save(newImgPath + imgFile)

    # Save label
    newXmlPath = f'./data/light/label/train/{className}/'
    if not os.path.exists(newXmlPath):
        os.makedirs(newXmlPath)
    newXmlPath = newXmlPath + fileName + '.xml'
    changeLabel(label, newXmlPath, newImgPath, bboxes)


# In[ ]:


import xml.etree.ElementTree as ET
from os import getcwd
import glob

def convert_annotation(annotation_voc, train_all_file):
    tree = ET.parse(annotation_voc)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1: continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        train_all_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

train_all_file = open('./data/light/train_all.txt', 'w')

# Get annotations_voc list
for className in classes:
    annotations_voc = glob.glob(f'./data/light/label/train/{className}/*.xml')
    for annotation_voc in annotations_voc:
        image_id = annotation_voc.split('/')[-1].split('.')[0]+'.JPG'
        train_all_file.write(f'./data/light/image/train/{className}/{image_id}')
        convert_annotation(annotation_voc, train_all_file)
        train_all_file.write('\n')
train_all_file.close()


# In[ ]:


# yolo.py 파일에 학습된 모델을 반환하는 함수
def get_model(self):
    return self.yolo_model

#학습이 완료된 모델을 저장 
model = yolo.get_model()
model.save('model_data/light_tiny_model.h5')


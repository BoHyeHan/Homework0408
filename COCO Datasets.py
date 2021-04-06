#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from object_detection.utils import label_map_util

category_index = label_map_util.create_category_index_from_labelmap('models/research/object_detection/data/mscoco_complete_label_map.pbtxt', use_display_name=True)
print(category_index)
detection_class = 0
name = category_index[detection_class + 1]['name']
print(name)


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
get_ipython().run_line_magic('matplotlib', 'inline')

##########모델 로드

category_index = label_map_util.create_category_index_from_labelmap('models/research/object_detection/data/mscoco_complete_label_map.pbtxt', use_display_name=True)

configs = config_util.get_configs_from_pipeline_file('ssd_mobilenet_v1_coco_2018_01_28/pipeline.config')
model = model_builder.build(model_config=configs['model'], is_training=False)

image_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
x_test, shapes = model.preprocess(image_tensor)

dict_output = model.predict(x_test, shapes)

dict_output = model.postprocess(dict_output, shapes)
detection_boxes = dict_output['detection_boxes'] 
detection_classes = dict_output['detection_classes'] 
detection_scores = dict_output['detection_scores'] 
num_detections = dict_output['num_detections'] 

sess = tf.compat.v1.Session()

saver = tf.compat.v1.train.Saver()
saver.restore(sess, 'ssd_mobilenet_v1_coco_2018_01_28/model.ckpt')

##########모델 예측

image = Image.open('car.jpg')
numpy_image = np.array(image)
x_test = np.array([numpy_image])

detection_boxes, detection_classes, detection_scores, num_detections = sess.run([detection_boxes,
                            detection_classes,
                            detection_scores,
                            num_detections], feed_dict={image_tensor: x_test})

detection_classes = detection_classes.astype(dtype=np.int32)
num_detections = num_detections.astype(dtype=np.int32)

viz_utils.visualize_boxes_and_labels_on_image_array(
    numpy_image,
    detection_boxes[0],
    detection_classes[0] + 1,
    detection_scores[0],
    category_index,
    instance_masks=None, 
    use_normalized_coordinates=True, 
    min_score_thresh=0.5)

plt.figure(figsize=(20, 20))
plt.imshow(numpy_image)
plt.show()


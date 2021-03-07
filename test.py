print('[test.py] Initiating script - Load Trained Model from Checkpoint')

import os
import tensorflow as tf
import cv2 
import numpy as np

from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
MODEL_PATH = WORKSPACE_PATH + '/models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

checkpointName = 'ckpt-21'
windowName = 'Gesture Recognition'

#

print('[test.py] Loading pipeline config and building a detection model')

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(
  model_config = configs['model'],
  is_training = False
)

#

print('[test.py] Restoring checkpoint')

ckpt = tf.compat.v2.train.Checkpoint(
  model = detection_model
)
ckpt.restore(os.path.join(CHECKPOINT_PATH, checkpointName)).expect_partial()

@tf.function
def detect_fn(image):
  image, shapes = detection_model.preprocess(image)
  prediction_dict = detection_model.predict(image, shapes)
  detections = detection_model.postprocess(prediction_dict, shapes)
  return detections

#

print('[test.py] Preparing capture for real time detection')

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#

print('[test.py] Loading window')

while True: 
  ret, frame = cap.read()
  image_np = np.array(frame)
  
  input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), 
    dtype = tf.float32
  )
  detections = detect_fn(input_tensor)
  
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates = True,
    max_boxes_to_draw = 5,
    min_score_thresh = .5,
    agnostic_mode = False
  )

  cv2.imshow(windowName, cv2.resize(image_np_with_detections, (800, 600)))
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    break

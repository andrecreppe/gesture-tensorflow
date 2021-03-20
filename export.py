print('[collector.py] Initiating script - Model exporter')

import os

WORKSPACE_PATH = 'Tensorflow/workspace'
APIMODEL_PATH = 'Tensorflow/models'
MODEL_PATH = WORKSPACE_PATH + '/models'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

#

print("[export.py] Exporting checkpoints")

exportCommand = """python {}/research/object_detection/exporter_main_v2.py --input_type=image_tensor --pipeline_config_path={}/{}/pipeline.config --trained_checkpoint_dir={} --output_directory={}export""".format(
  APIMODEL_PATH,
  MODEL_PATH,
  CUSTOM_MODEL_NAME,
  CHECKPOINT_PATH,
  CHECKPOINT_PATH
)

os.system(exportCommand)

#

print('[export.py] Converting exported model')

exportItems = 'detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores'

convertCommand = """tensorflowjs_converter --input_format=tf_saved_model --output_node_names='{}' --output_format=tfjs_graph_model --signature_name=serving_default {}export/saved_model {}converted""".format(
  exportItems,
  CHECKPOINT_PATH,
  CHECKPOINT_PATH
)

os.system(convertCommand)

print('[export.py] Finished exporting trained model!')

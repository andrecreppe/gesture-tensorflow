print('[train.py] Initiating script - Training model')

import os
import sys
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'

labelQtd = 5
labels = [
  {'name': 'Hello', 'id': 1},
  {'name': 'Yes', 'id': 2},
  {'name': 'No', 'id': 3},
  {'name': 'Thanks', 'id': 4},
  {'name': 'I Love You', 'id': 5}
]

trainSteps = 20000

#

print('[train.py] Creating Label Map')

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
  for label in labels:
    f.write('item { \n')
    f.write('\tname:\'{}\'\n'.format(label['name']))
    f.write('\tid:{}\n'.format(label['id']))
    f.write('}\n')

#

print('[train.py] Creating tfrecords')

trainRecordCommand = '{} -x {} -l {} -o {}'.format(
  SCRIPTS_PATH + '/generate_tfrecord.py',
  IMAGE_PATH + '/train',
  ANNOTATION_PATH + '/label_map.pbtxt',
  ANNOTATION_PATH + '/train.record'
)

testRecordCommand = '{} -x {} -l {} -o {}'.format(
  SCRIPTS_PATH + '/generate_tfrecord.py',
  IMAGE_PATH + '/test',
  ANNOTATION_PATH + '/label_map.pbtxt',
  ANNOTATION_PATH + '/test.record'
)

os.system('python {}'.format(trainRecordCommand))
os.system('python {}'.format(testRecordCommand))

# 

print('[train.py] Updating Config For Transfer Learning')
print("[train.py] Make sure to have copied the 'pipeline.config' as stated in the README")

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, 'r') as f:
  proto_str = f.read()
  text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = labelQtd
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, 'wb') as f:
  f.write(config_text)

#

def query_yes_no(question, default = 'yes'):
  valid = {'yes': True, 'y': True, 'ye': True,
            'no': False, 'n': False}
  if default is None:
    prompt = ' [y/n] '
  elif default == 'yes':
    prompt = ' [Y/n] '
  elif default == 'no':
    prompt = ' [y/N] '
  else:
    raise ValueError("invalid default answer: '%s'" % default)

  while True:
    sys.stdout.write(question + prompt)
    choice = input().lower()
    if default is not None and choice == '':
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write("Please respond with 'yes' or 'no' "
                      "(or 'y' or 'n').\n")

print('[train.py] Maps and records ready!')
runnow = query_yes_no('[train.py] Do you want to execute the training now?')

runTrainingCommand = 'python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps={}'.format(
  APIMODEL_PATH,
  MODEL_PATH,
  CUSTOM_MODEL_NAME,
  MODEL_PATH,
  CUSTOM_MODEL_NAME,
  trainSteps
)

if runnow:
  os.system(runTrainingCommand)
  print('[train.py] Finished training model')
else:
  print('[train.py] Execute this command later in the project folder:')
  print('> ' + runTrainingCommand)

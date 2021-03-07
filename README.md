# signlanguage-tensorflow
Real Time Sign Language Detection with Tensorflow Object Detection and Python | Deep Learning SSD

To run again
- Collector.ipynb
- Run labelImg and label
- Copy 13 files to train and 2 to test

To install the object_detection API you need to follow a [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation)

-------------------------------------

# Download TF Models Pretrained Models from Tensorflow Model Zoo
!cd Tensorflow && git clone https://github.com/tensorflow/models

# Copy Model Config to Training Folder

!mkdir {'Tensorflow\workspace\models\\' + CUSTOM_MODEL_NAME}
!cp {PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PATH + '/' + CUSTOM_MODEL_NAME}

--------------------------

Resources used:
- wget.download('https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py')
- Setup https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
#https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation
- wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
- !mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz {PRETRAINED_MODEL_PATH}
- !cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

This code was based on the tutorial from [Nicholas Renotte](https://www.youtube.com/watch?v=ZTSRZt04JkY) on YouTube.

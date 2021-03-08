# gesture-tensorflow

Real Time Sign Language Detection with Tensorflow Object Detection and Python

![Execution example](https://github.com/andrecreppe/gesture-tensorflow/blob/main/images/example.png?raw=true)

---

## 🐍 Installation

This project requires [Tensorflow](https://www.tensorflow.org/install/pip?hl=pt-br), [OpenCV](https://pypi.org/project/opencv-python/) and Tensorflow Object Detection packages in order to run properly.

The first two you can install directly using *pip* (or *conda* if you prefer):

```bash
$ pip install tensorflow
$ pip install opencv-python
```

To install the last, [object_detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation), you need to follow the tutorial in the highlighted link. It walks through the process of setting up the package and also the **Tensorflow-GPU** in order gain a lot of performance during runtime (highly recommended).

Also, you need to Download TensorFlow's Pretrained Models from [Tensorflow Model Zoo](https://github.com/tensorflow/models)

```bash
$ cd Tensorflow
$ git clone https://github.com/tensorflow/models
```

After that, you need to copy the Model Config (*pipeline.config*) from the pre-trained folder (*ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8*) to the actual Training Folder. You will need to create it under the *../workspace/models/* directory, and the model name is *my_ssd_mobnet*.

```bash
$ mkdir Tensorflow/workspace/models/my_ssd_mobnet
$ cp Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config Tensorflow/workspace/models/my_ssd_mobnet
# copy this {Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config} to {Tensorflow/workspace/models/my_ssd_mobnet}
```

### 🖍️ labelImg

In order to label the collected images you need to install the necessary packages to make this software work.

You can find the installation requirements by checking their [GitHub page](https://github.com/tzutalin/labelImg) or at *./labelImg/README.md* inside this repository.

## 🎥 Executing

**First** step: get some images to train.

To do so, you need to edit some variables in the *collector* script:
- Add your desired labels in the *labels* array
- Set how many images you want (int) to collect in *imgQtd*

After editing, you can run the following command:

```bash
$ python collector.py
```

**Second** step: select the pose region using [labelImg](https://github.com/tzutalin/labelImg)

In order to label the gestures inside our collected photos, you need to use the labelImage software. 

Select with the labeling tool the desired area (for example, your hand with the chosen 'Hello' gesture) and write the correct label name (this name is going to appear during testing, so beware of capitalization such as 'hello' or 'Hello')

Here are some stats for using this software:
- Important configs
  - Enable *Autosave mode* in the View tab
  - Open Dir > ../workspace/images/collectedImages
  - Save Dir > ../workspace/images/collectedImages (the same as before)
- Quick Commands
  - W = open selection
  - A = back
  - D = forward

To run labelImg you can execute:

```bash
$ python labelImg/labelImg.py
```

**Third** step: training our model

To execute the training script, you need to make modifications to some variables inside the *train* script:
- Set how many (int) labels you are going to train in *labelQtd*
- Add the labels exactly as you named them using *labelImg* (beware of capitalization) to the *labels* object
  - Do not forget to sequence the ID's. The order does not matter here
- Set how many (int) train steps you are going to perform in *trainSteps*
  - 10000 is ok
  - 20000 is more precise

This step can take some time. Using my GTX 1070 Zotac with 8GB the training duration for each step was about
- ~16min in 10000 steps 
- ~30min in 20000 steps

After doing the modifications, you can run the following command:

```bash
$ python train.py
```

**Fourth** step: testing our trained model

The only variable that you need to change for the recognition to work is the *checkpointName* with the value for the last checkpoint generated during model training. 

To get this name you need to navigate to *Tensorflow/workspace/models/my_ssd_mobnet* and there get the highest checkpoint value (example: *ckpt-21*).

After editing it, you can finally run the *test* script:

```bash
$ python test.py
```

## 📚 Source

This code was based on the tutorial from [Nicholas Renotte](https://youtu.be/pDXdlXlaCco) on YouTube.

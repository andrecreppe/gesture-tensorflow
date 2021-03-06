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

After labeling the collected images, you will need to sort some images from */collectedImages* to the outside folders: */train* and */test*.
- You need to copy the image with the *.xml* generated by labelImg
- Do not copy, just sort in some proportions. For 15 collected images, you can:
  - Move 13 to */train*
  - Move 2 to */test*

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

## 📤 Exporting

In order to export the trained model for other uses (like applying the trained model to a [React application](https://github.com/andrecreppe/gesture-tensorflow-react)) you can run the *export* script:
- Make sure to include in the *exportItems* variable the desired parameters.

```bash
# Required package for exporting to a NodeJS application
$ pip install tensorflowjs

# Export to JS application script
$ python export.py
```

Once finished, the exported and converted files can be found at the */converted* folder inside the model folder (in this case, *my_ssd_mobnet*).

### 🌎 Hosting

A good method for storing the converted weights is using a cloud service. For this project the recommended one is [IBM Cloud](https://cloud.ibm.com/) due to the easy setup process and good free tier.

After signing in/up, you need to create a **Cloud Object Store** (free tier option) and then a **Bucket**. Then, upload the entire */converted* folder content, probably 4 files such as:
- group1-shard1of3.bin
- group1-shard2of3.bin
- group1-shard3of3.bin
- model.json

The next step is to go and make the now populated Bucket public. That option is available in the *Bucket Access Policies > Public Access*.

Now is necessary to enable the CORS functionality for the Bucket. The easiest to do it is using the [IBM Cloud CLI](https://github.com/IBM-Cloud/ibm-cloud-cli-release/releases) After installing, you can now work with the cmd/terminal. 

The first thing to do is to make a sign in into your IBM Cloud Account by prompting the following command:

```bash
$ ibmcloud login
```

Make sure to be in the same region as the created Bucke (check that on the web page). If you need to change it, do it by running the target command: 

```bash
# change 'us-south' for your bucket region
$ ibmcloud target -r us-south
```

Before proceeding, make sure to have installed the [IBM Cloud Object Storage Plugin](https://github.com/IBM/ibmcloud-object-storage-plugin) by executing it's installation command:

```bash
$ ibmcloud plugin install cloud-object-storage
```

Now we need to enable the CORS (Cross-Origin Resource Sharing) for our bucket in order to access the uploaded files from any application anywhere. 

The necessary configuration modifications can be found inside the *corsconfig.json* file, and to apply it to our Bucket you need to run, inside the project's root folder, this command to upload the CORS configuration JSON file:

```bash
$ ibmcloud cos bucket-cors-put --bucket BUCKETNAME --cors-configuration file://corsconfig.json
```

After completing these steps, our trained model is online and fully operational. To access them in other applications, you can use the *Object's Public URL*, provided when you click on the file inside *Bucket > Objects* at the IBM Cloud Web page.

## 📚 Source

This code was based from the follwing **Nicholas Renotte** tutorials on YouTube:
- [Real Time Sign Language Detection with Tensorflow Object Detection and Python | Deep Learning SSD](https://youtu.be/pDXdlXlaCco)
- [Building a Real Time Sign Language Detection App with React.JS and Tensorflow.JS | Deep Learning](https://youtu.be/ZTSRZt04JkY)

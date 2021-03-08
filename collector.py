print('[collector.py] Initiating script - Image Collector')

import cv2 #opencv
import os
import time
import uuid

IMG_PATH = 'Tensorflow/workspace/images/collectedImages'

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
imgQtd = 15

#

print('[collector.py] Preparing capture')

for label in labels:
  newdirpath = os.path.join('Tensorflow', 'workspace', 'images', 'collectedImages', label)

  if not os.path.exists(newdirpath):
    print('Created new directory: {}' + newdirpath)
    os.mkdir(newdirpath)
  
  cap = cv2.VideoCapture(0)
  print('Collecting images for {}'.format(label))
  time.sleep(5)

  count = 1
  
  for imgnum in range(imgQtd):
    ret, frame = cap.read()
    imgname = label + '.' + '{}.jpg'.format(str(uuid.uuid1()))
    newimgpath = os.path.join(IMG_PATH, label, imgname)
    
    frameName = '{} - Frame {}'.format(label, count) 

    cv2.imwrite(newimgpath, frame)
    cv2.imshow('Frame', frame)
    time.sleep(2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    count += 1
  
  cap.release()
  cv2.destroyAllWindows()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import random
import warnings
from cProfile import label
import cvlib as cv
from cvlib.object_detection import draw_bbox

warnings.filterwarnings("ignore")

print('Image(Train) :',len(os.listdir('training_images')))
print('Image(Test) :',len(os.listdir('testing_images')))

Data=pd.read_csv('train_solution_bounding_boxes (1).csv')

Data.head()

print('Training Data : ',len(Data))

for i in Data.values:
  photo=plt.imread(f'training_images\{i[0]}')
  #plt.imshow(photo)
  print('Photo shape:',photo.shape)
  print('Name,xmin,ymin,xmax,ymax:',i)
  pt1=(int(i[1]),int(i[2]))
  pt2=(int(i[3]),int(i[4]))
  color=(255, 0, 0)
  thickness = 2
  cv2.rectangle(photo,pt1,pt2, color, thickness)
  plt.figure()
  #plt.imshow(photo)
  break

#Compter le nombre de voiture
x=0
for a,i in enumerate(Data.values):

  img=plt.imread(f'training_images\{i[0]}')

  bbox, label, conf = cv.detect_common_objects(img)
  output_image = draw_bbox(img, bbox, label, conf)

  #plt.imshow(output_image)

  sum = label.count('car')
  x = x + sum

  if a==5:
    break
print('Le nombre de voiture est '+str(x))

for a,i in enumerate(Data.values):
  img=plt.imread(f'training_images\{i[0]}')
  print(img.shape)
  plt.figure()
  #plt.imshow(img)
  xmin=int(i[1])
  ymin=int(i[2])
  xmax=int(i[3])
  ymax=int(i[4])
  cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(255, 0, 0),2)
  plt.figure()
  #plt.imshow(img)
  if a ==2:
    break

cv2.setUseOptimized(True) # Optimisation
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() # Selective search object
im = cv2.imread(f'training_images/vid_4_1000.jpg')
im=cv2.resize(im,(224,224))
plt.figure()
#plt.imshow(im)
ss.setBaseImage(im) # on charge l'image
ss.switchToSelectiveSearchFast() # Recherche selective
rects = ss.process()
print('Shape:',im.shape)
print('possible bounty boxes:',len(rects))

for rect in rects:
  x, y, w, h = rect
  imOut=cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
plt.figure()
#plt.imshow(imOut);

#Faire le train de toutes les images

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
def get_iou(bb1, bb2):

    assert bb1['x1'] < bb1['x2'] #bb1
    assert bb1['y1'] < bb1['y2']

    assert bb2['x1'] < bb2['x2'] #bb2
    assert bb2['y1'] < bb2['y2'];

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
      return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

image_liste=[]
k=0
l=0
z=0
for a in pd.read_csv('train_solution_bounding_boxes (1).csv').values:
  Name,xmin,ymin,xmax,ymax=a
  bb1={ 
            'x1':int(xmin),
            'y1':int(ymin),
            'x2':int(xmax),
            'y2':int(ymax)
            }
  try:
    img=cv2.imread('training_images/'+Name)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    for i in rects:
      x, y, w, h = i # On selectionne les meilleurs images
      bb2={'x1':x, 
          'y1':y,
          'x2':x+w,
          'y2':y+h
          }
      img1=img[bb2['y1']:bb2['y2'],bb2['x1']:bb2['x2']] # Rogner l'image
      img1_shape=cv2.resize(img1,(224,224))
      if k<l:
            if 0.5<get_iou(bb1,bb2):  
              image_liste.append([img1_shape,1])
              k+=1
      else:
        if 0.5<get_iou(bb1,bb2):  
          image_liste.append([img1_shape,1])
          k+=1
        else:
          image_liste.append([img1_shape,0])
          l+=1
  except Exception as e:
    print('Erreur ',e)
  z+=1
  print(Name,z,len(rects))

data=[]
data_label=[]
for features,label in image_liste:
  data.append(features)
  data_label.append(label)
print('Transfert réussie')

print('Nombre de photos : ',len(data),
'|Label : ',len(data_label))

len(image_liste)

data=[]
data_label=[]
for features,label in image_liste:
  data.append(features)
  data_label.append(label)
print('ok')


i=random.randint(1,10502)
print('Classe : ',data_label[i])
print('Taille image : ',data[i].shape)
#plt.imshow(data[i]);

data=np.asarray(data)
data_label=np.asarray(data_label)

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(data,data_label,test_size=0.33, random_state=42)

base_model=tf.keras.applications.VGG16(include_top=False,input_shape=(224,224,3),weights='imagenet')

model=tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

base_model.trainable=False
for i,layer in enumerate(base_model.layers):
  print(i,layer.name,'-',layer.trainable)
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics='accuracy')

epoch=4
hist=model.fit(x_train,y_train,epochs=epoch,validation_data=(x_val,y_val))



#TESTING 
car=[]
tentative_img=cv2.imread(f'testing_images/vid_5_26580.jpg')
ss.setBaseImage(tentative_img)
ss.switchToSelectiveSearchFast()
rects1 = ss.process()
print('Nombre de voiture possible : ',len(rects1))
for i in rects1:
  x, y, w, h = i
  bb3={'x1':x,
        'y1':y,
        'x2':x+w,
        'y2':y+h
      }
  try:
    assert bb3['x1'] < bb3['x2']
    assert bb3['y1'] < bb3['y2']
    img_data=tentative_img[bb3['y1']:bb3['y2'],bb3['x1']:bb3['x2']]
    img_data=cv2.resize(img_data,(224,224))
    predict=model.predict(img_data.reshape(1,224,224,3))
    if predict[0]>0.5:
      car.append([bb3,predict[0]])
    else:
      pass
  except Exception as e:
    print('Erreur ',e)
print('Nombre de image avec une prédiction de classe 1 :',len(car))
print('-------------------------------------------------------------------------')
tentative_img=cv2.imread(f'testing_images/vid_5_26580.jpg')
car[np.argmax(np.array(car)[:,1])][0]
pt1=(car[np.argmax(np.array(car)[:,1])][0]['x1'],car[np.argmax(np.array(car)[:,1])][0]['y1'])
pt2=(car[np.argmax(np.array(car)[:,1])][0]['x2'],car[np.argmax(np.array(car)[:,1])][0]['y2'])

plt.figure()
plt.imshow(tentative_img)
cv2.rectangle(tentative_img,pt1,pt2,(255, 0, 0),2)
plt.figure()
plt.imshow(tentative_img);
plt.show()




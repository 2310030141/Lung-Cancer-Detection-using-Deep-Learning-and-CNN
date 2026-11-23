import tensorflow as tf
import cv2
import numpy as np
import os

model = tf.keras.models.load_model("lung_cancer_model.h5")

classes = ["Benign","Malignant","Normal"]

folder = "ct_scans/test_cases"

for img_name in os.listdir(folder):

    path = os.path.join(folder,img_name)

    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    print(img_name,"->",classes[np.argmax(pred)])
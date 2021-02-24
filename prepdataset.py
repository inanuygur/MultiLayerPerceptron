import os
import math
import gzip
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# kodun bulundugu adrese dair bilgiyi cekmek icin
real_filepath = os.path.realpath(__file__)
parent_dir = Path(real_filepath).parent
print('parent_dir: ', parent_dir)

''' ------------------------------------------------ '''
# Egitim veri setini yukleme
f = gzip.open(str(parent_dir) + "\\train-images-idx3-ubyte.gz",'r')

image_size = 28
num_images = 60000

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

train_images = []
for i in range(0, 60000):
  image = np.asarray(data[i]).squeeze()
  train_images.append(image)


f = gzip.open(str(parent_dir) + "\\train-labels-idx1-ubyte.gz",'r')
f.read(8)
train_labels = []
for i in range(0, 60000):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    train_labels.append(labels[0])


# Test Veri setinin yukleme
f = gzip.open(str(parent_dir) + "\\t10k-images-idx3-ubyte.gz",'r')

image_size = 28
num_images = 10000

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

test_images = []
for i in range(0, 10000):
  image = np.asarray(data[i]).squeeze()
  test_images.append(image)


f = gzip.open(str(parent_dir) + "\\t10k-labels-idx1-ubyte.gz",'r')
f.read(8)
test_labels = []
for i in range(0, 10000):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    test_labels.append(labels[0])

train_images = np.array(train_images)
test_images = np.array(test_images)

''' ------------------------------------------------ '''
# Yukarida '*.zip' halde bulunan veri setleri cekiliyor 
# ve numpy arrat haline getirilerek dosya dizinine kaydediliyor.
# Ana kodda bu numpy dizileri cekiliyor ve egitim sirasinda 
# ve sonrasinda kullaniliyor. 

''' ------------------------------------------------ '''
#One Hot Encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
#Normalizing train values
train_images = np.array(train_images)
train_images = train_images/255.0
test_images = np.array(test_images)
test_images = test_images/255.0

''' ------------------------------------------------ '''
print('train_images.shape:', train_images.shape)
print('train_labels.shape:', train_labels.shape)
print('test_images.shape:', test_images.shape)
print('test_labels.shape:', test_labels.shape)

''' ------------------------------------------------ '''
# # Ana kodda deneme yapabilmek icin bu kod blogunu calistirmak ve numpy arraylerini elde etmeniz gerekiyor <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# np.save(str(parent_dir) + "\\numpies\\train_images", train_images)
# np.save(str(parent_dir) + "\\numpies\\train_labels", train_labels)
# np.save(str(parent_dir) + "\\numpies\\test_images", test_images)
# np.save(str(parent_dir) + "\\numpies\\test_labels", test_labels)

''' ------------------------------------------------ '''
plt.imshow(test_images[5])
print(test_labels[5])
plt.show()

''' ------------------------------------------------ '''
# KOD SONU # 

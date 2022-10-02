import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import datasets,layers,models
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, ReLU

#from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.keras.optimizers import SGD

#from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix


#mnistデータの読み込み、tensorflow.kerasで用意されている
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
#訓練用：60000, テスト用：10000
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))


# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0


#アーキテクチャの構築
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


#モデルのコンパイルと学習
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start=time.time()
model.fit(train_images, train_labels, epochs=5)
end=time.time()

#モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'計算時間：{end-start}')
print(f'精度：{test_acc}')

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import keras

"""## 2.Read data and do data cleaning"""

df = pd.read_csv("Fake_News_Dataset/train.csv")
df.head(5)

df=df.dropna()

df.groupby('label').describe()

"""## 3.Create training data and testing data"""

df_true = df[df['label']==0]
df_true.shape

df_fake = df[df['label']==1]
df_fake.shape

df_true_downsampled = df_true.sample(df_fake.shape[0])
df_true_downsampled.shape

df_balanced = pd.concat([df_true_downsampled, df_fake])
df_balanced.shape

df_balanced['label'].value_counts()

df_balanced.columns

from sklearn.model_selection import train_test_split
# stratify means you can split data with portion
X_train, X_test, y_train, y_test = train_test_split(df_balanced['title'],df_balanced['label'], train_size=0.9 , stratify=df_balanced['label'])

"""## 4.Choose Bert model from tensorflow hub """

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1, name='dropout')(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
    return tf.keras.Model(text_input, net)

model = build_classifier_model()

"""## 5.Plot model architecture"""

tf.keras.utils.plot_model(model)

"""## 6.Train the model"""

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=tf.keras.metrics.BinaryAccuracy(name='accuracy'))

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

"""## 7.Plot the training process"""

import matplotlib.pyplot as plt

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

"""## 8.Save the model"""

!mkdir -p saved_model
model.save('saved_model/Fake-News-Detection-Model-With-Bert')
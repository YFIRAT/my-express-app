# yf_step_1
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# yf_step_2
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Veriyi normalleştirin (0-255 arasındaki piksel değerlerini 0-1 arasına çekiyoruz)
train_images = train_images / 255.0
test_images = test_images / 255.0


# yf_step_3
from tensorflow.keras import layers, models
from tensorflow.keras import Input

model = models.Sequential()

# İlk katman olarak Input kullan
model.add(Input(shape=(28, 28, 1)))

# Daha sonra diğer katmanları ekleyebilirsin
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# yf_step_4
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# yf_step_5
model.fit(train_images, train_labels, epochs=5)


# yf_step_6
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')



# yf_step_7
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Veri setini yükle
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Veriyi ölçeklendirme (0-255 arası piksel değerlerini 0-1 arası değerler haline getirme)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Etiketleri kategorik hale getirme
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Modeli eğitme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)



# yf_step_7
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test doğruluğu:', test_acc)



# yf_step_8
import matplotlib.pyplot as plt

# Eğitim geçmişini kaydetme
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Eğitim ve doğrulama kayıplarını grafikte gösterme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.show()





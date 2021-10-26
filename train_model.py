import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Dinh nghia
data_path  = "dataset_landscape"
image_size  = 64 # Resize de tiet kiem thoi gian train
n_epochs = 64
n_batchsize = 32

# Load anh tu thu muc data
def load_normal_images(data_path):
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path  in normal_images_path:
        full_img_path = os.path.join(data_path, img_path)
        img = image.load_img(full_img_path, target_size=(image_size, image_size), color_mode="grayscale")
        img = image.img_to_array(img)
        img = img/255
        # Dua vao list
        normal_images.append(img)
    normal_images = np.array(normal_images)
    return normal_images

# Ham tao nhieu ngau nhien
def make_noise(normal_image):
    w, h, c = normal_image.shape
    mean = 0
    sigma = 1
    gauss = np.random.normal(mean, sigma, (w, h, c))
    gauss = gauss.reshape(w, h, c)

    noise_image = normal_image + gauss * 0.08
    return noise_image

# Ham tao tap du lieu noise
def make_noise_images(normal_images):
    noise_images = []
    for img in normal_images:
        noise_image = make_noise(img)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images

# How show thu du lieu
def show_imageset(imageset):
    f, ax = plt.subplots(1, 5)
    for i in range(1,6):
        ax[i-1].imshow(imageset[i].reshape(64,64), cmap="gray")
    plt.show()

# Tao model Auto Encoder
def make_ae_model():
    input_img = Input(shape=(image_size, image_size, 1), name='image_input')

    # encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool2')(x)

    # decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2, 2), name='upsample1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2, 2), name='upsample2')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Conv5')(x)

    # model
    model = Model(inputs=input_img, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model



# Tao model
denoise_model = make_ae_model()
denoise_model.summary()

if not os.path.exists('data.dat'):
    # Load anh normal
    normal_images = load_normal_images(data_path)
    # Tao anh noise
    noise_images = make_noise_images(normal_images)

    # Chia du lieu train test
    noise_train, noise_test, normal_train, normal_test = train_test_split(noise_images, normal_images, test_size=0.2)
    with open("data.dat", "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
else:
    with open("data.dat", "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]


# Train model
early_callback = EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
denoise_model.fit(noise_train, normal_train, epochs=n_epochs, batch_size=n_batchsize,
                  validation_data=(noise_test, normal_test),
                  callbacks=[early_callback])

denoise_model.save("denoise_model.h5")

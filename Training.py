import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Gerekli kütüphaneleri import etme
path = "C:\\Users\\90537\\Desktop\\Deneme\\myData"
# Tüm sınıf klasörlerinin bulunduğu dizin

labelFile = 'labels.csv'
# Sınıf isimlerinin bulunduğu dosya

batch_size_val=50
# Bir seferde işlenecek veri sayısı

epochs_val=50
# Modelin eğitileceği epoch sayısı

imageDimesions = (32,32,3)
# Görüntü boyutları (genişlik, yükseklik, renk kanalı)

testRatio = 0.2
# Test verisinin oranı (%20)

validationRatio = 0.2
# Doğrulama verisinin oranı (%20)

# Görüntülerin ve sınıfların import edilmesi
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
# Tespit edilen toplam sınıf sayısı
sinif_sayisi = len(myList)
print("Importing Classes.....")
# Sınıfların import edilmesi
for x in range(0, len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Veriyi bölme işlemi
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Eğitim ve test veri setlerinin şekillerinin kontrol edilmesi
print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "Eğitim setindeki görüntü sayısı etiket sayısına eşit değil"
assert(X_validation.shape[0] == y_validation.shape[0]), "Doğrulama setindeki görüntü sayısı etiket sayısına eşit değil"
assert(X_test.shape[0] == y_test.shape[0]), "Test setindeki görüntü sayısı etiket sayısına eşit değil"
assert(X_train.shape[1:] == (imageDimesions)), "Eğitim görüntülerinin boyutları yanlış"
assert(X_validation.shape[1:] == (imageDimesions)), "Doğrulama görüntülerinin boyutları yanlış"
assert(X_test.shape[1:] == (imageDimesions)), "Test görüntülerinin boyutları yanlış"

# CSV dosyasını okuma
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# Her sınıftan bazı örnek görüntülerin gösterilmesi
num_of_samples = []
cols = 5
num_classes = sinif_sayisi
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# Her kategorideki örnek sayısını gösteren çubuk grafik
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Eğitim veri setinin dağılımı")
plt.xlabel("Sınıf numarası")
plt.ylabel("Görüntü sayısı")
plt.show()

# Görüntülerin gri tonlamalı hale getirilmesi işlemi
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Görüntülerin histogram eşitleme işlemi
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# Görüntülerin ön işleme işlemi
def preprocessing(img):
    # Görüntünün gri tonlamalı hale getirilmesi
    img = grayscale(img)
    # Görüntünün histogramının eşitlenmesi
    img = equalize(img)
    # Görüntünün normalleştirilmesi (0 ile 1 arasında ölçeklendirme)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])

# Derinlik boyutunun eklenmesi
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Görüntülerin artırılması (augmentasyon)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# Artırılmış görüntü örneklerini gösterme
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

# Etiketlerin kategorik hale getirilmesi
y_train = to_categorical(y_train, sinif_sayisi)
y_validation = to_categorical(y_validation, sinif_sayisi)
y_test = to_categorical(y_test, sinif_sayisi)

# Konvolüsyonel sinir ağı modeli
def myModel():
    #kullanılacak filtre sayısı
    no_Of_Filters = 60
    # kullanılacak filtre boyutu
    size_of_Filter = (5, 5)
    # kullanılacak filtre boyutu
    size_of_Filter2 = (3, 3)
    #havuzlamana katmanındaki havuz boyutu
    size_of_pool = (2, 2)
    #düğüm sayısı
    no_Of_Nodes = 500
    # keras sequential modeli oluşturur.
    model = Sequential()
    # Modelin ilk katmanı olarak bir konvolüsyon katmanı eklenir. Bu katman, belirtilen filtre sayısı ve boyutu ile birlikte relu aktivasyon fonksiyonu kullanır.
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    # ikinci bir konvolüsyon katmanı eklenir. Giriş şekli artık belirtilmez, çünkü ilk katmanda belirtilmiştir.
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    # Maksimum havuzlama katmanı eklenir.
    model.add(MaxPooling2D(pool_size=size_of_pool))
    # Daha fazla konvolüsyon ve havuzlama katmanları eklenir.(eğitimin verimi artsın diye.)
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    #düzleştirme katmanı 2 boyutlu matrisi 1 boyutlu matrise dönderiyor.
    model.add(Flatten())
    # Tam bağlantılı bir gizli katman eklenir.
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    # Çıkış katmanı eklenir. Bu katman, sınıf sayısı kadar çıkış düğümüne sahiptir ve softmax aktivasyon fonksiyonu kullanır.
    model.add(Dense(sinif_sayisi, activation='softmax'))
    # Model derlenir. Adam optimizer kullanılır, kayıp fonksiyonu olarak categorical crossentropy seçilir ve doğruluk metriği 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modeli eğitme
model = myModel()
print(model.summary())
history = model.fit(x=X_train, y=y_train, batch_size=batch_size_val, epochs=epochs_val, validation_data=(X_validation, y_validation), shuffle=True)

# Eğitim sonuçlarını görselleştirme
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Modelin test verisi üzerindeki değerlendirmesi
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Modelin pickle objesi olarak kaydedilmesi
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)

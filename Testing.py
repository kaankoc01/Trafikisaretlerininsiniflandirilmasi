import cv2
import numpy as np
import pickle
import pyttsx3

# Yüz tanıma için Cascade Classifier'ı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# EĞİTİLMİŞ MODELİ İÇE AKTAR
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Sınıf isimlerini global scope'ta tanımla
class_names = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
    'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
    'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    return class_names[int(classNo)]  # classNo'yu tamsayıya dönüştür

# Ses sentezini başlat
engine = pyttsx3.init()

# Kameradan görüntü yakalamak için
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare yakala
    success, img_original = cap.read()
    if not success:
        break

    # Yüz tanıma
    faces = face_cascade.detectMultiScale(img_original, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:  # Eğer yüz bulunamadıysa
        # Görüntüyü işle
        img = cv2.resize(img_original, (32, 32))  # Yeniden boyutlandırma işlemini burada gerçekleştirin
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        # Tahmin yap
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)[0]  # classIndex'i bir tamsayıya dönüştür
        probabilityValue = np.amax(predictions)

        # Sınıf adını ve olasılığı ekrana yazdır
        if probabilityValue > 0.90 and classIndex in range(len(class_names)):  # Belirli bir güven eşiği belirle ve trafik levhası sınıflarını kontrol et
            sinif_adi = getClassName(classIndex)
            cv2.putText(img_original, f"{sinif_adi} ({probabilityValue*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tanımlanan levhanın etrafına dikdörtgen çiz
            height, width, _ = img_original.shape
            cv2.rectangle(img_original, (5, 5), (width-5, height-5), (0, 255, 0), 2)
            
            # Seslendirme yap
            engine.say(sinif_adi)   
            engine.runAndWait()

    # Görüntüyü göster
    cv2.imshow("Result", img_original)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import tensorflow as tf
import numpy as np
import cv2

def mnist_data_process():
    #mnist veri yolu belirleniyor.
    path = 'mnist.npz'
    #mnist veri seti tanımlanıyor.
    mnist = tf.keras.datasets.mnist

    #tanımlanan veri seti yükleniyor.
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    #fonksiyon oluşturulan veri setini döndürüyor.
    return (x_train, y_train, x_test, y_test)


#Model eğitim fonksiyonu:
def model_train_process(x_train, y_train, x_test, y_test):
    #Epoch sonrası doğruluk değeri kontrolü için feedback sınıfı
    #ve bu sınıfa bağlı epoch_process fonksiyonu.
    #Fonksiyonun amacı parametre olarak aldığı log fonksiyonundan accuracy değerinin kontrolünü sağlamak.
    #Accuracy (doğruluk) değeri 0.99'dan büyükse yani doğruluk oranı %99'dan fazla ise modeli eğitmeyi sonlandırıyor.
    class feedback(tf.keras.callbacks.Callback):
        def epoch_process(self, epoch, logs={}):
            print(logs)
            if (logs.get('accuracy') > 0.99):
                print("\nProcess canceled because training accuracy is 99%!")
                self.model.stop_training = True

    #feedback sınıfına bağlı feedbacks nesnesi oluşturuluyor.
    feedbacks = feedback()
    #Fotoğraf değerleri (opencv'den okunan fotoğraf değerleri) 255'e bölünüyor.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #Modeli sıralı ve ardışık olarak tanımlıyoruz. (Keras sequential sıralı ve ardışık olarak model arayüzü tanımlamamı sağlıyor.)
    model = tf.keras.models.Sequential([
        #Son katmana gidecek verileri 28x28 olarak şekillendirip gönderiyorum.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        #Dense sınıfı sonraki katmanlara veri iletimini sağlarken:
            #relu bizlere negatif bölgedeki tanjantları 0 varsayarak işlem hızını daha da arttırıyor.
            #tabii ki bu karar vermede hatalara sebep olsa da işlemi hızlandırıyor. Bu proje için yeterli bir ölçüttür.
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
            #softmax regresyon mnist veri setinde sıkça kullanılan öğrenme algoritmasıdır.
            #Burada da dense sayesinde bir önceki katmandan çıkan sonucun (output) softmax algoritmasına bağlanmasını sağlamış olduk.
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    #Optimizer classs:
        # Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam
    #Loss Functions:
        # Sparse Categorical Cross Entropy, Huber, Hinge, Categorical Hinge, Binary Cross Entropy, KLDivergence, LogCosh
    #Metrics:
        # Accuracy, Binary Accuracy, Mean, Mean Absolute Error, AUC, Binary Accuracy Entropy
    model.compile(optimizer='adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    #Model eğitim log.
    history = model.fit(x_train, y_train, epochs=10, callbacks=[feedbacks])
    print(history.epoch, history.history['accuracy'][-1])
    return model

#tahmin fonksiyonu.
def prediction_process(model, img):
    #Parametredeki fotoğraf verisini dizi halinde saklar.
    imgs = np.array([img])
    #oluşturulan dizi verisini model için recursive olarak uygular.
    res = model.predict(imgs)
    #modelin en büyük çıktılarının değerini index olarak atar.
    index = np.argmax(res)
    #index değişkenini fonksiyonun sonucu olarak sunar.
    return str(index)

#Kameradan alınan görüntüyü binary değerlere çevirmek için önceden tanımladığım eşikleme değeri.
threshold = 100

def webcam_process(model):
    #global threshold eşikleme değeri tanımlandı.
    global threshold
    #0 portundan kameraya erişim sağlandı.
    cap = cv2.VideoCapture(0)
    #açılan kamera ekranının ismi belirlendi.
    frame = cv2.namedWindow('Number Detector')
    #ekran boyutu ayarlandı.
    background = np.zeros((480, 640), np.uint8)

    #filtreleme için genel bir değişken oluşturuldu.
    frameCount = 0

    #bir döngü içerisinde kameradan gelen görüntünün eş zamanlı olarak modele iletilip görüntüdeki el yazısının tespiti sağlandı.
    while True:
        #kamera ekranından okumalar yapılıyor.
        ret, frame = cap.read()

        #kamera ekranı filtreleniyor. (siyah ve beyaz bir kamera ekranı için)
        frameCount += 1
        frame[0:480, 0:80] = 0
        frame[0:480, 560:640] = 0
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Kameradaki görüntü binary yani siyah ve beyaz olarak yeniden şekillendiriliyor.
        _, thresholdRes = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

        #Kamera ekranındaki çerçeve ve arka plan ayarlanıyor.
        #Hesaplamalar açılan ekran boyutuna göre belirleniyor.
        resizedFrame = thresholdRes[240 - 75:240 + 75, 320 - 75:320 + 75]
        background[240 - 75:240 + 75, 320 - 75:320 + 75] = resizedFrame

        #Kamerada önceden belirlenen çerçeveden kareler fotoğraf olarak kaydedilip model ve tahmin fonksiyonlarına gönderiliyor.
        iconImg = cv2.resize(resizedFrame, (28, 28))

        #kameradan alınan fotoğraf ve oluşturulup eğitilen modele göre sayı tahmini predictResult değişkenine atanıyor.
        predictResult = prediction_process(model, iconImg)

        #frame sayısına göre ekran filtreleniyor.
        if frameCount == 5:
            background[0:480, 0:80] = 0
            frameCount = 0

        #Ekranın sol üst tarafına tahmin fonksiyonundan gelen sonucu yazdırıyorum.
        #Kamera ekranında bir çerçeve oluşturuluyor.
        cv2.putText(background, predictResult, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.rectangle(background, (320 - 80, 240 - 80), (320 + 80, 240 + 80), (255, 255, 255), thickness=3)

        cv2.imshow('Number Detector', background)

        #Kapatma butonuna basıldığında açılan kamera ekranı kapatılıyor.
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break

    #kamera ve açılan kamera ekranı kapatılıyor.
    cap.release()
    cv2.destroyAllWindows()

model = None
try:
    #Modelin yüklenmesi deneniyor. Eğer yüklenirse daha önceden eğitilmiş demektir. Yüklenmezse mnist veri seti kullanılarak model eğitiliyor.
    model = tf.keras.models.load_model('hand-writing-model.sav')
    print('Model is already trained. Just loading...')
    #Modele dair istatistikler yazdırılıyor.
    print(model.summary())
except:
    #Mnist veri seti yükleniyor.
    (x_train, y_train, x_test, y_test) = mnist_data_process()
    #Daha önce oluşturulan model mnist verisine göre eğitiliyor.
    model = model_train_process(x_train, y_train, x_test, y_test)
    #Eğitilen model .sav olarak kaydediliyor.
    model.save('hand-writing-model.sav')
    print("Saved hand writing model !")

#Kamera başlatılıyor
webcam_process(model)
import cv2
import sys

# получить данные от пользователя
imagePath = 'g20.jpg'
cascPath = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

# читаем изображение
img = cv2.imread(imagePath)

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # определить лицо на фотографии
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30, 30)
    )

    # рисовать квадратик вокруг лица
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.квадрат(картинка, (координаты), (где рисуем), цвет, толщина_линии)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imshow('Faces found', img)
cv2.waitKey(0)


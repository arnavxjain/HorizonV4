"""
@name: `HorizonV4`
@author: Arnav Jain
@dateOfCreation: 03/09/2021
"""
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

print('Loading modules...')

path = 'assets'
images = []
classNames = []
myList = os.listdir(path)

for each in myList:
    newImg = cv2.imread(f'{path}/{each}')
    images.append(newImg)
    classNames.append(os.path.splitext(each)[0])


def findEncodings(images):
    encodedImages = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodedImages.append(encoded)

    return encodedImages


def log(name):
    with open('log.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(', ')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


print('Running encoding function')
encodedImages = findEncodings(images)
print('Encoding complete')

print('Returning webcam parameters')
cap = cv2.VideoCapture(0)

while True:
    success, imgL = cap.read()
    img = cv2.resize(imgL, (0, 0), None, 0.25, 0.25)
    imgMain = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(img)
    encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)

    for encodeFace, faceLocation in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodedImages, encodeFace)

        faceDis = face_recognition.face_distance(encodedImages, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(imgMain, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(imgMain, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
            cv2.putText(imgMain, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            log(name)

    cv2.imshow('HorizonV4 Engine', imgMain)
    cv2.waitKey(1)
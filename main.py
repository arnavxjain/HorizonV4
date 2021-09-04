"""
@name: `HorizonV4`
@author: Arnav Jain
@dateOfCreation: 03/09/2021
"""
import os
import cv2
import numpy as np
import face_recognition

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


print('Running encoding function')
encodedImages = findEncodings(images)
print('Encoding complete')

print('Returning webcam parameters')
cap = cv2.VideoCapture(0)

while True:
    success, imgL = cap.read()
    img = cv2.resize(imgL, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(img)
    encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)

    for encodeFace, faceLocation in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodedImages, encodeFace)

        faceDis = face_recognition.face_distance(encodedImages, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]


# # Loading assets[0]
# print('Loading Assets...')
# epo1 = face_recognition.load_image_file("assets/epo1.jpg")
# epo1 = cv2.cvtColor(epo1, cv2.COLOR_BGR2RGB)
#
# # Loading assets[1]
# print('Coloring the assets')
# epo2 = face_recognition.load_image_file("assets/epo2.jpg")
# epo2 = cv2.cvtColor(epo2, cv2.COLOR_BGR2RGB)
#

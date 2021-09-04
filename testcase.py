import cv2
import numpy as np
import face_recognition

print('Loading modules...')

# Loading assets[0]
print('Loading Assets...')
epo1 = face_recognition.load_image_file("assets/epo1.jpg")
epo1 = cv2.cvtColor(epo1, cv2.COLOR_BGR2RGB)

# Loading assets[1]
print('Coloring the assets')
epo2 = face_recognition.load_image_file("assets/epo2.jpg")
epo2 = cv2.cvtColor(epo2, cv2.COLOR_BGR2RGB)

# Reading face light points and locating face
print('Encoding and loading face')
epo1Location = face_recognition.face_locations(epo1)[0]
encoded_epo1 = face_recognition.face_encodings(epo1)[0]
cv2.rectangle(epo1, (epo1Location[3], epo1Location[0]), (epo1Location[1], epo1Location[2]), (0, 155, 0), 2)

epo2Location = face_recognition.face_locations(epo2)[0]
encoded_epo2 = face_recognition.face_encodings(epo2)[0]
cv2.rectangle(epo2, (epo2Location[3], epo2Location[0]), (epo2Location[1], epo2Location[2]), (0, 155, 0), 2)

result = face_recognition.compare_faces([encoded_epo1], encoded_epo2)[0]
distance = face_recognition.face_distance([encoded_epo1], encoded_epo2)

print('Initializing Module Presentation')
cv2.imshow("HorizonV4 - EPO1", epo1)
cv2.imshow("HorizonV4 - EPO2", epo2)
cv2.waitKey(0)

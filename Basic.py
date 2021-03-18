import cv2
import numpy as np
import face_recognition

# Thêm ảnh
imgLuyn = face_recognition.load_image_file('imageBasic/obama.jpeg')
imgLuyn = cv2.cvtColor(imgLuyn, cv2.COLOR_BGR2RGB)
imgLuyn = cv2.resize(imgLuyn, (0,0), fx=0.5, fy=0.5)
# Thêm ảnh test để so sánh
imgLuynTest = face_recognition.load_image_file('imageBasic/biden.jpeg')
imgLuynTest = cv2.cvtColor(imgLuynTest, cv2.COLOR_BGR2RGB)
imgLuynTest = cv2.resize(imgLuynTest, (0,0), fx=0.5, fy=0.5)

faceLocation = face_recognition.face_locations(imgLuyn)[0]
encodeLuyn =  face_recognition.face_encodings(imgLuyn)[0]
cv2.rectangle(imgLuyn,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]), (255,0,255),2)

faceLocationTest = face_recognition.face_locations(imgLuynTest)[0]
encodeLuynTest = face_recognition.face_encodings(imgLuynTest)[0]
cv2.rectangle(imgLuynTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]), (255,0,255),2)

result = face_recognition.compare_faces([encodeLuyn], encodeLuynTest)
faceDistance = face_recognition.face_distance([encodeLuyn], encodeLuynTest)
print(result, faceDistance)
cv2.putText(imgLuynTest, f'{result} {round(faceDistance[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255) , 2)

#
# cv2.imshow('Luyn', imgLuyn)
# cv2.imshow('LuynTest', imgLuynTest)
#
# cv2.waitKey(0)
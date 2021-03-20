import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'imageBasic'
images = []
className = []
myList = os.listdir(path)
# print(myList)

#Lấy tên phía trước của ảnh để in ra lúc mà điểm danh trong ggSheet
for cls in myList:
    #Chia ảnh thành 2 phần tên và đuôi
    # currentImage = cv2.imread(f'{path}/{cls}')
    currentImage = face_recognition.load_image_file(f'{path}/{cls}')
    images.append(currentImage)
    #lấy phần tử tên tương đương với phần tử đầu tiên của ảnh = 0
    className.append(os.path.splitext(cls)[0])
print(className)

# Mã hoá các ảnh để làm data so sánh
def findEncondings(images):
    #Tạo mảng rỗng đểm thêm ảnh sau khi đã
    encodeList = []
    for img in images:
        #encode là xuất ra ảnh BGR nên mình phải convert sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        #Thêm ảnh đã convert vào mảng rỗng đã tạo ở trên
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncondings(images)
# print(len(encodeListKnown))
print('Encoding Complete')

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            currentTime = datetime.now()
            time = currentTime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

#tạo webcam
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Location
    facesCurrentFrameLocation = face_recognition.face_locations(imgS)
    #Encode
    faceEncodeCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrameLocation)
    # Để camera nhận diện được khuôn mặt
    for face_encoding, face_loc in zip(faceEncodeCurrentFrame, facesCurrentFrameLocation):
        matches = face_recognition.compare_faces(encodeListKnown,face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        matchIndex = np.argmin(face_distances)
        #Nếu đã có ảnh train
        if matches[matchIndex]:
            nameOfPeople = className[matchIndex].upper()
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1,y2), (x2,y1), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35) , (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, nameOfPeople,(x1, y2), cv2.FONT_HERSHEY_COMPLEX,1 ,(255,255,255) , 2 )
        #Nếu k có ảnh
        else:
            nameOfPeople = 'unknown'.upper()
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1, y2), (x2, y1), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, nameOfPeople, (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        #sử dụng hàm điểm danh
        markAttendance(nameOfPeople)
    #mở webcam
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

train
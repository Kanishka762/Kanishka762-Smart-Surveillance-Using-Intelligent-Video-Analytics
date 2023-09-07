"""Importing required libraries............."""

import cv2
import os
import face_recognition
import numpy as np
from ultralytics import YOLO

"""Uploaded image path"""

path = "/home/srihari/deepstreambackend/test/face_test/face"

# """Loading YOLO face detection model globally"""

# model = YOLO("./yoloface.pt")


# """Face detection function creation which returns bbox coordinates in required format"""
# def facedetection(image_path):
#     results = model.predict(image_path)
#     coords = []

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             print(box.cls)
#             b = box.xyxy[0]
#             c = tuple(b.numpy())
#             d = []
#             for i in c:
#                 i = int(i)
#                 d.append(i)
#             first_element = d.pop(0)
#             d.append(first_element)
#             coords.append(tuple(d))
#     return coords


"""Generating class names and encodings......"""

images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencodings(images)
print("Number of identities in DB-1: ", len(encodelistknown))


"""Initiating face recognition for incoming faces in the frame"""

def recognition(imgs):
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facecurframe = face_recognition.face_locations(imgs)
    # facecurframe = facedetection(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    for encodeface, faceloc in zip(encodecurframe, facecurframe):
        print(faceloc)
        y1, x2, y2, x1 = faceloc
        cv2.rectangle(imgs, (x1,y1), (x2,y2), (0,255,0), 3)

        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        faceids = face_recognition.face_distance(encodelistknown,encodeface)
        if min(faceids) <= 0.54:
            matchindex = np.argmin(faceids) 

            if matches[matchindex]:
                name = classnames[matchindex].upper()
                print(name)
                cv2.rectangle(imgs, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(imgs, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)

            else:
                for encodeface, faceloc in zip(encodecurframe, facecurframe):
                    cv2.rectangle(imgs, (x1,y1), (x2,y2), (0,255,0), 3)
                    cv2.putText(imgs, "Unknown", (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)

        else:
            for encodeface, faceloc in zip(encodecurframe, facecurframe):
                cv2.rectangle(imgs, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(imgs, "Unknown", (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),2)
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    # cv2.imshow("face-recognition-window", imgs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cap = cv2.VideoCapture("/home/srihari/deepstreambackend/test/face_test/ab.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   

result = cv2.VideoWriter('filename1.avi', 
                         cv2.VideoWriter_fourcc(*'DIVX'),
                         25, size)
while True:
    _ , imgs = cap.read()
    recognition(imgs)
    result.write(imgs)
    # cv2.imshow("cam", imgs)
    if cv2.waitKey(1) == ord('q'):
        break
result.release()
cap.release()
# cv2.destroyAllWindows()
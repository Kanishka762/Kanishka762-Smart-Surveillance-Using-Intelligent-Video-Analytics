from io import BytesIO
import face_recognition 
import subprocess as sp
import cv2
from datetime import datetime  #datetime module to fetch current time when frame is detected
import numpy as np
from pytz import timezone 
from nanoid import generate
from ultralytics import YOLO

import random
# from main import gen_datainfo
import uuid


face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

model = YOLO("./models_deepstream/best.pt")

# datainfo = gen_datainfo()
# print(datainfo)
# def generate_random_id(length=18):
#     alphabets = list('abcdefghijklmnopqrstuvwxyz')

#   # Generate a random ID having the specified length.
#     random_id = ''.join([random.choice(alphabets) for _ in range(length)])
#     return random_id


def facedetection(im):
    results = model.predict(im)
    coords = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = tuple(b.numpy())
            d = []
            for i in c:
                i = int(i)
                d.append(i)
            first_element = d.pop(0)
            print(first_element)
            d.append(first_element)
            coords.append(tuple(d))
    return coords

def find_person_type(im0,datainfo):
    print(im0)
    cv2.imwrite("/home/agx123/deepstreambackend/face_test.jpg",im0)
    known_whitelist_faces = datainfo[0]
    known_blacklist_faces = datainfo[1]
    unknown_faces = []
    unknown_id = []
    known_whitelist_id = datainfo[2]
    known_blacklist_id = datainfo[3]
    minimum_distance = []
    np_arg_src_list = known_whitelist_faces + known_blacklist_faces
    np_bytes2 = BytesIO()
    # np.save(np_bytes2, im0, allow_pickle=True)
    # np_bytes2 = np_bytes2.getvalue()
    # image = cv2.imread(im0) # if im0 does not work, try with im1
    image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(image)
    # locations = facedetection(image)
    # print(locations)
    encodings = face_recognition.face_encodings(image,locations)

    did = None
    track_type = "100"

    if len(locations) != 0:
        if len(known_whitelist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_white = face_recognition.compare_faces(known_whitelist_faces,face_encoding)
                faceids_white = face_recognition.face_distance(known_whitelist_faces,face_encoding)
                matchindex_white = np.argmin(faceids_white)
                if min(faceids_white) <=0.70:
                    if matches_black[matchindex_white]:
                    
                        did = '00'+ str(known_whitelist_id[matchindex_white])
                        track_type = "00"

        if len(known_blacklist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_black = face_recognition.compare_faces(known_blacklist_faces,face_encoding)
                faceids_black = face_recognition.face_distance(known_blacklist_faces,face_encoding)
                matchindex_black = np.argmin(faceids_black)
                print(faceids_black)
                
                
                if min(faceids_black) <=0.70:
                    if matches_black[matchindex_black]:
                    
                        did = '01'+ str(known_blacklist_id[matchindex_black])
                        track_type = "01"

        if len(unknown_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                matchindex_unknown = np.argmin(faceids_unknown)
                minimum_distance.append(min(faceids_unknown))
                if min(faceids_unknown) <=0.70:
                    if matches_unknown[matchindex_unknown]:
                    
                        did = "11" + str(unknown_id[matchindex_unknown])
                        track_type = "11"

        else:
            unknown_faces.append(encodings)
            id = str(uuid.uuid4())
            did = id
            track_type = "10"
            if id not in unknown_id:
                unknown_id.append(id)



    return did, track_type

# im0 = cv2.imread("/home/nivetheni/TCI_express_srihari/TCI_express/image/QmWzcNaQTmrUsaswdaTibkyuc3HYTCSfsr9wTtjNtfhfEq.jpg")
# print(find_person_type(im0,datainfo))

    
    
    
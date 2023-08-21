from io import BytesIO
import face_recognition 
import subprocess as sp
import cv2
from datetime import datetime  #datetime module to fetch current time when frame is detected
import numpy as np
from pytz import timezone 
from nanoid import generate
from ultralytics import YOLO
import torch
import random
# from main import gen_datainfo
import uuid
from modules.face_recognition_pack.lmdb_list_gen import insertWhitelistDb, insertBlacklistDb, insertUnknownDb
import random, string
#/home/srihari/deepstreambackend/modules/face_recognition_pack/facedatainsert_lmdb.py

face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

# model = YOLO("./models_deepstream/best.pt")

whitelist_faces = []
whitelist_ids = []
blacklist_faces = []
blacklist_ids = []
unknown_faces = []
unknown_ids = []




def randomword():
    length = 7 
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def load_lmdb_list():
    whitelist_faces1, whitelist_id1 = insertWhitelistDb()
    blacklist_faces1, blacklist_id1 = insertBlacklistDb()
    unknown_faces1, unknown_id1 = insertUnknownDb()

    global whitelist_faces
    whitelist_faces = whitelist_faces1
    global whitelist_ids
    whitelist_ids = whitelist_id1

    global blacklist_faces
    blacklist_faces = blacklist_faces1
    global blacklist_ids
    blacklist_ids = blacklist_id1

    global unknown_faces
    unknown_faces = unknown_faces1
    global unknown_ids
    unknown_ids = unknown_id1
    print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))

def facedetection(im):
    results = model.predict(im)
    coords = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu()
            c = tuple(b.numpy())
            d = []
            for i in c:
                i = int(i)
                d.append(i)
            first_element = d.pop(0)
            # # print(first_element)
            d.append(first_element)
            coords.append(tuple(d))
    return coords

def most_common_element(input_list):
    if not input_list:
        return None
    
    max_count = 0
    most_common = None
    
    for element in input_list:
        count = input_list.count(element)
        if count > max_count:
            max_count = count
            most_common = element
    
    return most_common

def faceDecisionmaker(faceDecisionDictionary, didRefer):
    max_key = max(faceDecisionDictionary, key=faceDecisionDictionary.get)
    did = most_common_element(didRefer[max_key])
    return did , max_key

def find_person_type(listOfCrops):
    # # print("len(listOfCrops)",len(listOfCrops))
    faceDecisionDictionary = {}
    didRefer = {}
    for oneCrop in listOfCrops:
        try:
            did, track_type = faceRecognition(oneCrop[0])
        except:
            pass
        # if not faceRecognition(oneCrop[0]):
        #     # print(oneCrop[0].shape)
        #     cv2.imwrite("/home/srihari/deepstreambackend/error_imgs_crops/"+randomword()+".jpg",oneCrop[0])


        if track_type in faceDecisionDictionary:
            faceDecisionDictionary[track_type] = faceDecisionDictionary[track_type] + 1
        else:
            faceDecisionDictionary[track_type] = 1
        if track_type in didRefer:
            didRefer[track_type].append(did)
        else:
            didRefer[track_type] = []
            didRefer[track_type].append(did)
    if '' in faceDecisionDictionary:
        removed_value = faceDecisionDictionary.pop('')
    # # print(faceDecisionDictionary)
    did, track_type = faceDecisionmaker(faceDecisionDictionary, didRefer)
    return did , track_type


def faceRecognition(im0):

    global whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids
    # # print(len(whitelist_ids), len(blacklist_ids), len(unknown_ids), whitelist_ids, blacklist_ids, unknown_ids)
    minimum_distance = []
    np_arg_src_list = whitelist_faces + blacklist_faces
    np_bytes2 = BytesIO()
    # np.save(np_bytes2, im0, allow_pickle=True)
    # np_bytes2 = np_bytes2.getvalue()
    # image = cv2.imread(im0) # if im0 does not work, try with im1
    image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(image)
    # locations = facedetection(image)
    # # print(locations)
    # # print(len(locations))
    for location in locations:
        y1, x2, y2, x1 = location
        crop = image[y1:y2, x1:x2]

        # cv2.imwrite("/home/agx123/deepstreambackend/face_test.jpg",crop)
    
    encodings = face_recognition.face_encodings(image,locations)
    
    did = ""
    track_type = "100"

    if len(locations) != 0:

        if len(whitelist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_white = face_recognition.compare_faces(whitelist_faces,face_encoding)
                faceids_white = face_recognition.face_distance(whitelist_faces,face_encoding)
                matchindex_white = np.argmin(faceids_white)
                if min(faceids_white) <=0.70:
                    if matches_white[matchindex_white]:
                    
                        did = '00'+ str(whitelist_ids[matchindex_white])
                        track_type = "00"
                        # print("++++++++++++++++++++++++++++++++++++")
                        print(did)
                        print(track_type)
                        # print("++++++++++++++++++++++++++++++++++++")
                        return did, track_type


        if len(blacklist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_black = face_recognition.compare_faces(blacklist_faces,face_encoding)
                faceids_black = face_recognition.face_distance(blacklist_faces,face_encoding)
                matchindex_black = np.argmin(faceids_black)
                # # print(faceids_black)
                
                
                if min(faceids_black) <=0.70:
                    if matches_black[matchindex_black]:
                    
                        did = '01'+ str(blacklist_ids[matchindex_black])
                        track_type = "01"
                        # print("--------------------------------------")
                        print(did)
                        print(track_type)
                        # print("--------------------------------------")
                        return did, track_type


        if len(unknown_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                matchindex_unknown = np.argmin(faceids_unknown)
                minimum_distance.append(min(faceids_unknown))
                # # print("faceids_unknown length is ",len(faceids_unknown[0])," for ","unknown_faces,face_encoding of length ", len(unknown_faces),len(face_encoding))
                if min(faceids_unknown) <=0.70:
                    if matches_unknown[matchindex_unknown]:
                        did = "11" + str(unknown_ids[matchindex_unknown])
                        track_type = "11"
                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        print(did)
                        print(track_type)
                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        return did, track_type

                else:
                    # print("encodings", encodings)
                    unknown_faces.append(encodings[0])
                    id = str(uuid.uuid4())
                    did = id
                    track_type = "10"
                    if id not in unknown_ids:
                        unknown_ids.append(id)
                        # print(len(whitelist_ids), len(blacklist_ids), len(unknown_ids), whitelist_ids, blacklist_ids, unknown_ids)

                        # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                        print(did)
                        print(track_type)
                        # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")

                        return did, track_type
        



        else:
            # print("encodings", encodings)

            unknown_faces.append(encodings[0])
            id = str(uuid.uuid4())
            did = id
            track_type = "10"
            if id not in unknown_ids:
                unknown_ids.append(id)
                # print(len(whitelist_ids), len(blacklist_ids), len(unknown_ids), whitelist_ids, blacklist_ids, unknown_ids)

                # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                print(did)
                print(track_type)
                # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                return did, track_type
    else:
        did = ""
        track_type = "100"
        return did, track_type

#/home/srihari/deepstreambackend/error_imgs_crops
# import os

# arr = os.listdir('/home/srihari/deepstreambackend/error_imgs_crops')
# for img_path in arr:
#     # print("/home/srihari/deepstreambackend/error_imgs_crops/"+img_path)
#     im0 = cv2.imread("/home/srihari/deepstreambackend/error_imgs_crops/"+img_path)
#     # print(faceRecognition(im0))

# im0 = cv2.imread("/home/nivetheni/TCI_express_srihari/TCI_express/image/QmWzcNaQTmrUsaswdaTibkyuc3HYTCSfsr9wTtjNtfhfEq.jpg")
# # print(find_person_type(im0))

    
    
    
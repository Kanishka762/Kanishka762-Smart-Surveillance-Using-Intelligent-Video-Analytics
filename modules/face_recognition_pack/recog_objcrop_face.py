from modules.components.load_paths import *
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
# from modules.face_recognition_pack.lmdb_list_gen import insertWhitelistDb, insertBlacklistDb, insertUnknownDb
from modules.face_recognition_pack.convertCID2encodings import convertMemData2encoding
import lmdb

import random, string, threading, time, json
#/home/srihari/deepstreambackend/modules/face_recognition_pack/facedatainsert_lmdb.py

face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

db_env = lmdb.open(lmdb_path+'/face-detection',
                max_dbs=10)
faceDataDictDB = db_env.open_db(b'faceDataDictDB', create=True)
# model = YOLO("./models_deepstream/best.pt")


# whitelist_faces = []
# whitelist_ids = []
# blacklist_faces = []
# blacklist_ids = []
# unknown_faces = []
# unknown_ids = []
# callGlobalvariable = True
faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}

# def callFaceLists():
#     whitelist_faces = []
#     whitelist_ids = []
#     blacklist_faces = []
#     blacklist_ids = []
#     unknown_faces = []
#     unknown_ids = []
#     return whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids

# def merge_global():
#     # while True:
#     print("inside merge_global")
#     global whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids
#     # whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids = whitelist_faces1, blacklist_faces1, unknown_faces1, whitelist_ids1, blacklist_ids1, unknown_ids1
#     print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))
#     print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))
#         # time.sleep(1)
def insertLMDB(db_txn, key,value):

    for category in value:
        if category in ["whitelist_faces","blacklist_faces","unknown_faces"]:
            convList = []
            for encodings in value[category]:
                print('before encoding',encodings)
                encodings = json.dumps(encodings.tolist())
                print('encodings',type(encodings))
                convList.append(encodings)
            value[category] = convList

    db_txn.put(key.encode(), json.dumps(value).encode())

def fetchLMDB(db_txn, key):
    value = db_txn.get(key.encode())
    if value is not None:
        data = json.loads(value.decode())
        for category in data:
            if category in ["whitelist_faces","blacklist_faces","unknown_faces"]:
                listOfNumpuArray = []
                for encodedData in data[category]:
                    numpyArray = np.array(json.loads(encodedData))
                    # print('numpyArray',type(numpyArray))
                    listOfNumpuArray.append(numpyArray)
                data[category] = listOfNumpuArray

        return data
    else:
        return None

def load_lmdb_fst(mem_data_queue):
    while True:
        try:
            # global faceData, callGlobalvariable
            mem_data = mem_data_queue.get()
            print("got data from queue starting member data insertion")
            with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")

            print("got face data", faceData)

            if faceData is None:
                faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}
            
            # callGlobalvariable = False
            # print("changed to False: ",callGlobalvariable)
            print(len(mem_data))
            # global whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids
            # print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))
            # print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))
            # with global_lock:
            i = 0
            for each_member in mem_data:

                print(each_member['member'][0]['type'])

                if each_member['member'][0]['type'] in ['whitelist','blacklist']:
                    i = i+1
                    # print(each_member)
                    encodings, class_type, memberId = convertMemData2encoding(each_member)
                    
                    print('encodings',encodings)

                    # merge_global()
                    # merge_global(whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids)
                    
                    if class_type == "whitelist" and encodings is not None:
                        faceData['whitelist_faces'].append(encodings)
                        faceData['whitelist_ids'].append(memberId)
                        print("***********************")
                        
                        # print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))
                        # print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))

                    if class_type == "blacklist" and encodings is not None:
                        faceData['blacklist_faces'].append(encodings)
                        faceData['blacklist_ids'].append(memberId)
                        print("***********************")

                        # print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))
                        # print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))

                    # if class_type == "unknown" and encodings is not None:
                    #     faceData['unknown_faces'].append(encodings)
                    #     faceData['unknown_ids'].append(memberId)
                    #     print("***********************")
            # print("faceData",faceData)
            with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
                insertLMDB(db_txn, "faceData", faceData)
            print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
            print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
            # time.sleep(3)
            # callGlobalvariable = True
            # print("changed to True: ",callGlobalvariable)

        except Exception as e:
            print(e)
            



def randomword():
    length = 7 
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# def load_lmdb_list():
#     whitelist_faces1, whitelist_id1 = insertWhitelistDb()
#     blacklist_faces1, blacklist_id1 = insertBlacklistDb()
#     unknown_faces1, unknown_id1 = insertUnknownDb()

#     print(len(whitelist_faces1),len(blacklist_faces1),len(unknown_faces1))
#     print(len(whitelist_id1),len(blacklist_id1),len(unknown_id1))

#     global whitelist_faces
#     # print("before",whitelist_faces)
#     whitelist_faces = whitelist_faces1
#     # print("after",whitelist_faces)

#     global whitelist_ids
#     whitelist_ids = whitelist_id1

#     global blacklist_faces
#     blacklist_faces = blacklist_faces1
#     global blacklist_ids
#     blacklist_ids = blacklist_id1

#     global unknown_faces
#     unknown_faces = unknown_faces1
#     global unknown_ids
#     unknown_ids = unknown_id1
#     print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))

#     print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))

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
    listOfDIDs = []
    for each in input_list:
        listOfDIDs.append(list(each.keys())[0])
    # print(listOfDIDs)
    if not listOfDIDs:
        return None
    
    max_count = 0
    most_common = None
    
    for element in listOfDIDs:
        count = listOfDIDs.count(element)
        if count > max_count:
            max_count = count
            most_common = element
    # print("most_common", most_common)
    for each in input_list:
        if list(each.keys())[0] == most_common:
            return each

def faceDecisionmaker(faceDecisionDictionary, didRefer):
    max_key = max(faceDecisionDictionary, key=faceDecisionDictionary.get)
    # print("max_key", max_key)
    # print("didRefer[max_key]",didRefer[max_key])
    did_dict = most_common_element(didRefer[max_key])

    # print("did_dict", did_dict)

    return did_dict , max_key

def find_person_type(listOfCrops):
    global unknown_faces, unknown_ids
    print("len(listOfCrops)",len(listOfCrops))
    faceDecisionDictionary = {}
    didRefer = {}

    for oneCrop in listOfCrops:
        # print("insideloop")
        # faceRecognition(oneCrop[0])

        # did, track_type, encodings = faceRecognition(oneCrop[0])
        # print(did, track_type)
        try:
            # print("sending to face recog")
            data = faceRecognition(oneCrop[0])
    
            did, track_type, encodings = data
            # print(did)

            if track_type in faceDecisionDictionary:
                faceDecisionDictionary[track_type] = faceDecisionDictionary[track_type] + 1
            else:
                faceDecisionDictionary[track_type] = 1
            if track_type in didRefer:
                didRefer[track_type].append({did:encodings})
            else:
                didRefer[track_type] = []
                didRefer[track_type].append({did:encodings})
        except Exception as e:
            print(e)
            pass


    if '' in faceDecisionDictionary:
        removed_value = faceDecisionDictionary.pop('')
        didRefer.pop('')
    if len(faceDecisionDictionary) > 1:
        if '100' in faceDecisionDictionary:
            removed_value = faceDecisionDictionary.pop('100')
            didRefer.pop('100')



    # print(faceDecisionDictionary)
    # print(didRefer)
    
    did_dict, track_type = faceDecisionmaker(faceDecisionDictionary, didRefer)
        
    # print("did_dict, track_type", did_dict, track_type)

    for key, value in did_dict.items():
        did = key
        encodings = value
    # print("did, track_type",did, track_type)
    # if encodings is not None:
    #     if track_type == '10':
    #         # print(type(encodings))
    #         # print(encodings)
    #         try:
    #             encodings = encodings[0]
    #             # print(faceData['unknown_faces'])
    #             faceData['unknown_faces'].append(encodings)
    #             # print("unknown_facesfaceData['unknown_faces'])
    #             faceData['unknown_ids'].append(did)
    #             # print('unknown_ids',faceData['unknown_ids'])
    #         except:
    #             print("error")

    # print("***********************************************")
    # print(did, track_type)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    return track_type+did, track_type


def faceRecognition(im0):
    # print("starting face recog")
    # print(db_env)
    # print(faceDataDictDB)
    
    with db_env.begin(db=faceDataDictDB) as db_txn:
        # print(db_txn)
        faceData = fetchLMDB(db_txn, "faceData")

    # print('faceData',faceData)
    if faceData:
        # print(faceData)
        whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids = faceData['whitelist_faces'],faceData['blacklist_faces'],faceData['unknown_faces'],faceData['whitelist_ids'],faceData['blacklist_ids'],faceData['unknown_ids']
        
        minimum_distance = []
        np_arg_src_list = whitelist_faces + blacklist_faces
        np_bytes2 = BytesIO()
        # np.save(np_bytes2, im0, allow_pickle=True)
        # np_bytes2 = np_bytes2.getvalue()
        # image = cv2.imread(im0) # if im0 does not work, try with im1
        image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        # print("converted the colour")
        locations = face_recognition.face_locations(image)
        # print(locations)
        # locations = facedetection(image)
        # # print(locations)
        # # print(len(locations))
        for location in locations:
            y1, x2, y2, x1 = location
            crop = image[y1:y2, x1:x2]

            # cv2.imwrite("/home/agx123/deepstreambackend/face_test.jpg",crop)
        
        
        did = ""
        track_type = "100"

        if len(locations) != 0:
            encodings = face_recognition.face_encodings(image,locations)
            # print("starting whitelist comparision")
            if len(whitelist_faces):
                for face_encoding ,face_location in zip(encodings, locations):
                    matches_white = face_recognition.compare_faces(whitelist_faces,face_encoding)
                    faceids_white = face_recognition.face_distance(whitelist_faces,face_encoding)
                    matchindex_white = np.argmin(faceids_white)
                    if min(faceids_white) <=0.70:
                        if matches_white[matchindex_white]:
                        
                            did = str(whitelist_ids[matchindex_white])
                            track_type = "00"
                            # print("++++++++++++++++++++++++++++++++++++")
                            # print(did)
                            # print(track_type)
                            # print("++++++++++++++++++++++++++++++++++++")
                            return [did, track_type, encodings]

            # print("starting blacklist comparision")
            if len(blacklist_faces):
                # print(' ',blacklist_faces)
                for face_encoding ,face_location in zip(encodings, locations):
                    # print("face_encoding ,face_location",face_encoding ,face_location)
                    matches_black = face_recognition.compare_faces(blacklist_faces,face_encoding)
                    # print('matches_black',matches_black)
                    faceids_black = face_recognition.face_distance(blacklist_faces,face_encoding)
                    # print('faceids_black',faceids_black)
                    matchindex_black = np.argmin(faceids_black)
                    # print('matchindex_black',matchindex_black)
                    
                    # print('min(faceids_black)',min(faceids_black))
                    if min(faceids_black) <=0.70:
                        # print('matches_black[matchindex_black]',matches_black[matchindex_black])
                        if matches_black[matchindex_black]:
                        
                            did = str(blacklist_ids[matchindex_black])
                            track_type = "01"
                            # print("--------------------------------------")
                            # print(did)
                            # print(track_type)
                            # print("--------------------------------------")
                            return [did, track_type, encodings]

            # print("starting unknownlist comparision")
            if len(unknown_faces):
                # print("entering unknown recog")
                for face_encoding ,face_location in zip(encodings, locations):
                    # print(face_encoding ,face_location)
                    matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                    faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                    # print(matches_unknown ,faceids_unknown)
                    matchindex_unknown = np.argmin(faceids_unknown)
                    minimum_distance.append(min(faceids_unknown))
                    # print(matchindex_unknown, minimum_distance)
                    # # print("faceids_unknown length is ",len(faceids_unknown[0])," for ","unknown_faces,face_encoding of length ", len(unknown_faces),len(face_encoding))
                    if min(faceids_unknown) <=0.70:
                        # print("entering IF")
                        # print(matches_unknown[matchindex_unknown])
                        # if matches_unknown[matchindex_unknown]:
                        did = str(unknown_ids[matchindex_unknown])
                        track_type = "11"
                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        # print(did)
                        # print(track_type)
                        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        return [did, track_type, encodings]
                    
                    else:
                        # print("starting unknown repeat comparision")

                        # print("entering else")

                        # print("encodings", encodings)
                        # unknown_faces.append(encodings[0])
                        id = str(uuid.uuid4())
                        did = id
                        track_type = "10"
                        if id not in unknown_ids:
                            # unknown_ids.append(id)
                            # print(len(whitelist_ids), len(blacklist_ids), len(unknown_ids), whitelist_ids, blacklist_ids, unknown_ids)

                            # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                            # print(did)
                            # print(track_type)
                            # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")

                            return [did, track_type, encodings]
            



            else:
                # print("starting unknown repeat comparision")

                # print("encodings", encodings)
                id = str(uuid.uuid4())
                did = id
                track_type = "10"
                if id not in unknown_ids:
                    # print(len(whitelist_ids), len(blacklist_ids), len(unknown_ids), whitelist_ids, blacklist_ids, unknown_ids)

                    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                    # print(did)
                    # print(track_type)
                    # print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
                    return [did, track_type, encodings]

        
        else:
            # print("starting unidentified")

            did = ""
            track_type = "100"
            # print(did, track_type, None)
            return [did, track_type, im0]
    else:
        did = ""
        track_type = "100"
        # print(did, track_type, None)
        return [did, track_type, im0]
    
#/home/srihari/deepstreambackend/error_imgs_crops
# import os

# arr = os.listdir('/home/srihari/deepstreambackend/error_imgs_crops')
# for img_path in arr:
#     # print("/home/srihari/deepstreambackend/error_imgs_crops/"+img_path)
#     im0 = cv2.imread("/home/srihari/deepstreambackend/error_imgs_crops/"+img_path)
#     # print(faceRecognition(im0))

# im0 = cv2.imread("/home/nivetheni/TCI_express_srihari/TCI_express/image/QmWzcNaQTmrUsaswdaTibkyuc3HYTCSfsr9wTtjNtfhfEq.jpg")
# # print(find_person_type(im0))

    
    
    
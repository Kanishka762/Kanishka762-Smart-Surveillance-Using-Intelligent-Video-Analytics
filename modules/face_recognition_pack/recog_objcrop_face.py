from modules.components.load_paths import *
from init import loadLogger
from io import BytesIO
import face_recognition 
import subprocess as sp
import cv2
from datetime import datetime  
import numpy as np
from pytz import timezone 
from nanoid import generate
from ultralytics import YOLO
import torch
import random

import uuid

from modules.face_recognition_pack.convertCID2encodings import convertMemData2encoding
import lmdb
import threading
import random, string, threading, time, json

face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

db_env = lmdb.open(lmdb_path+'/face-detection',
                max_dbs=10,max_spare_txns = 20)
faceDataDictDB = db_env.open_db(b'faceDataDictDB', create=True)

faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}

def insertLMDB(db_txn, key,value):
    print("inside with")

    for category in value:
        if category in ["whitelist_faces","blacklist_faces","unknown_faces"]:
            convList = []
            for encodings in value[category]:
                encodings = json.dumps(encodings.tolist())
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

def insertUnknown(faceData):

    with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
        insertLMDB(db_txn, "faceData", faceData)

    print("after updation after updation after updation after updation after updation after updation")
    print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
    print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
    print("after updation after updation after updation after updation after updation after updation")

def updateMemberType(each_member):
    try:
        logger.info("trying to update face")
        logger.info("data before updation")
        logger.info(each_member)
        updatedtype = each_member['member'][0]['type']
        memberId = each_member['member'][0]['memberId']
        
        with db_env.begin(db=faceDataDictDB) as db_txn:
            faceData = fetchLMDB(db_txn, "faceData")
            logger.info("fetched data from LMDB")


        if faceData:
            print("satisfied if faceData:")
            if updatedtype == "blacklist":
                print("satisfied if updatedtype == blacklist:")
                if memberId in faceData['whitelist_ids']:
                    print('memberid is in whitelist')   
                    index = faceData['whitelist_ids'].index(memberId)
                    faceData['blacklist_ids'].append(faceData['whitelist_ids'].pop(index))
                    faceData['blacklist_faces'].append(faceData['whitelist_faces'].pop(index))
                    threading.Thread(target = insertUnknown,args = (faceData,)).start()

            if updatedtype == "whitelist":
                print("satisfied if updatedtype == whitelist:")
                if memberId in faceData['blacklist_ids']:
                    print('memberid is in blacklist')
                    index = faceData['blacklist_ids'].index(memberId)
                    faceData['whitelist_ids'].append(faceData['blacklist_ids'].pop(index))
                    faceData['whitelist_faces'].append(faceData['blacklist_faces'].pop(index))
                    threading.Thread(target = insertUnknown,args = (faceData,)).start()
        logger.info("updated data")
    except Exception as e:
        logger.error("An error while trying to update face", exc_info=e)

def load_lmdb_fst(mem_data_queue):
    while True:
        try:
            logger.info("Starting to listen to members queue")
            mem_data = mem_data_queue.get()
            logger.info("got data from queue")
            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")

            logger.info("got face data")
            logger.info(faceData)

            if faceData is None:
                faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}
            i = 0
            if len(mem_data) == 1 and mem_data[0]['updated']:
                logger.info("got member data for updation")

                threading.Thread(target = updateMemberType,args = (mem_data[0],)).start()
            else:
                logger.info("got member data for insertion")

                for each_member in mem_data:

                    if each_member['updated']:
                        threading.Thread(target = updateMemberType,args = (each_member,)).start()
                    
                    if each_member['member'][0]['type'] in ['whitelist','blacklist']:
                        i = i+1
                        encodings, class_type, memberId = convertMemData2encoding(each_member)

                        if class_type == "whitelist" and encodings is not None:
                            logger.info("updating whitelist")
                            faceData['whitelist_faces'].append(encodings)
                            faceData['whitelist_ids'].append(memberId)

                        if class_type == "blacklist" and encodings is not None:
                            logger.info("updating blacklist")
                            faceData['blacklist_faces'].append(encodings)
                            faceData['blacklist_ids'].append(memberId)

                        # if class_type == "unknown" and encodings is not None:
                        #     faceData['unknown_faces'].append(encodings)
                        #     faceData['unknown_ids'].append(memberId)
                
                threading.Thread(target = insertUnknown,args = (faceData,)).start()

        except Exception as e:
            logger.error("An error occurred during inserting face data into LMDB", exc_info=e)
            
def randomword():
    length = 7 
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

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

    if not listOfDIDs:
        return None
    
    max_count = 0
    most_common = None
    
    for element in listOfDIDs:
        count = listOfDIDs.count(element)
        if count > max_count:
            max_count = count
            most_common = element

    for each in input_list:
        if list(each.keys())[0] == most_common:
            return each

def faceDecisionmaker(faceDecisionDictionary, didRefer):
    max_key = max(faceDecisionDictionary, key=faceDecisionDictionary.get)
    did_dict = most_common_element(didRefer[max_key])
    return did_dict , max_key

def find_person_type(listOfCrops):
    global unknown_faces, unknown_ids
    print("len(listOfCrops)",len(listOfCrops))
    faceDecisionDictionary = {}
    didRefer = {}

    for oneCrop in listOfCrops:

        try:

            data = faceRecognition(oneCrop[0])
    
            did, track_type, encodings = data

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

    did_dict, track_type = faceDecisionmaker(faceDecisionDictionary, didRefer)
        

    for key, value in did_dict.items():
        did = key
        encodings = value

    if encodings is not None:
        if track_type == '10':
            print(type(encodings))
            print(encodings)
            try:
                with db_env.begin(db=faceDataDictDB) as db_txn:
                    faceData = fetchLMDB(db_txn, "faceData")
                print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
                print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
                encodings = encodings[0]

                faceData['unknown_faces'].append(encodings)

                faceData['unknown_ids'].append(did)
                print("updating unknown faces")
                print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
                print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
                print(db_env)
                print(faceDataDictDB)
                print(db_txn)
                threading.Thread(target = insertUnknown,args = (faceData,)).start()
                print('completed insertion')
                

            except Exception as e:
                print("error", e)

    print("***********************************************")
    print(did, track_type)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    return track_type+did, track_type, encodings

def faceRecognition(im0):
    
    with db_env.begin(db=faceDataDictDB) as db_txn:
        faceData = fetchLMDB(db_txn, "faceData")
    print("inside faceRecognition(im0)")
    
    if faceData:
        print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
        print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
        whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids = faceData['whitelist_faces'],faceData['blacklist_faces'],faceData['unknown_faces'],faceData['whitelist_ids'],faceData['blacklist_ids'],faceData['unknown_ids']
        
        minimum_distance = []
        np_arg_src_list = whitelist_faces + blacklist_faces
        np_bytes2 = BytesIO()
        image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        locations = face_recognition.face_locations(image)
        for location in locations:
            y1, x2, y2, x1 = location
            crop = image[y1:y2, x1:x2]

        did = ""
        track_type = "100"

        if len(locations) != 0:
            randd = randomword()
            encodings = face_recognition.face_encodings(image,locations)
            if len(whitelist_faces):
                for face_encoding ,face_location in zip(encodings, locations):
                    matches_white = face_recognition.compare_faces(whitelist_faces,face_encoding)
                    faceids_white = face_recognition.face_distance(whitelist_faces,face_encoding)
                    matchindex_white = np.argmin(faceids_white)
                    if min(faceids_white) <=0.70:
                        if matches_white[matchindex_white]:
                        
                            did = str(whitelist_ids[matchindex_white])
                            track_type = "00"
                            return [did, track_type, crop]

            if len(blacklist_faces):
                for face_encoding ,face_location in zip(encodings, locations):
                    matches_black = face_recognition.compare_faces(blacklist_faces,face_encoding)
                    faceids_black = face_recognition.face_distance(blacklist_faces,face_encoding)
                    matchindex_black = np.argmin(faceids_black)
                    if min(faceids_black) <=0.70:
                        if matches_black[matchindex_black]:
                        
                            did = str(blacklist_ids[matchindex_black])
                            track_type = "01"
                            return [did, track_type, crop]
            if len(unknown_faces):
                for face_encoding ,face_location in zip(encodings, locations):
                    matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                    faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                    matchindex_unknown = np.argmin(faceids_unknown)
                    minimum_distance.append(min(faceids_unknown))
                    if min(faceids_unknown) <=0.70:
                        did = str(unknown_ids[matchindex_unknown])
                        track_type = "11"
                        return [did, track_type, crop]
                    else:
                        id = str(uuid.uuid4())
                        did = id
                        track_type = "10"
                        if id not in unknown_ids:
                            return [did, track_type, crop]
            else:
                id = str(uuid.uuid4())
                did = id
                track_type = "10"
                if id not in unknown_ids:
                    return [did, track_type, crop]
        else:
            did = ""
            track_type = "100"
            return [did, track_type, im0]
    else:
        did = ""
        track_type = "100"
        return [did, track_type, im0]

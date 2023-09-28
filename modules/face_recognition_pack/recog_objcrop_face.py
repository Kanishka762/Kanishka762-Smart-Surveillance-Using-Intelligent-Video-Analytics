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


logger = loadLogger()

face_did_encoding_store = {}
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

db_env = lmdb.open(
    f'{lmdb_path}/face-detection', max_dbs=10, max_spare_txns=20
)
faceDataDictDB = db_env.open_db(b'faceDataDictDB', create=True)

faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}

def randomword():
    length = 7
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

class lmdboperationsFaceRecognition:
    def __init__(self):
        pass

    def insertLMDB(self,db_txn, key,value):
        print("inside with")

        for category in value:
            if category in ["whitelist_faces","blacklist_faces","unknown_faces"]:
                convList = []
                for encodings in value[category]:
                    encodings = json.dumps(encodings.tolist())
                    convList.append(encodings)
                value[category] = convList

        db_txn.put(key.encode(), json.dumps(value).encode())

    def fetchLMDB(self,db_txn, key):
        value = db_txn.get(key.encode())
        if value is None:
            return None
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

    def insertUnknown(self,faceData):

        with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
            self.insertLMDB(db_txn, "faceData", faceData)

        print("after updation after updation after updation after updation after updation after updation")
        print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
        print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
        print("after updation after updation after updation after updation after updation after updation")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class memberQueueService:
    def __init__(self):
        self.faceLmdbOPSobj = lmdboperationsFaceRecognition()

    def updateMemberType(self,each_member):
        try:
            logger.info("trying to update face")
            logger.info("data before updation")
            logger.info(each_member)
            updatedtype = each_member['member'][0]['type']
            memberId = each_member['member'][0]['memberId']

            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = self.faceLmdbOPSobj.fetchLMDB(db_txn, "faceData")
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
                        threading.Thread(target = self.faceLmdbOPSobj.insertUnknown,args = (faceData,)).start()

                elif updatedtype == "whitelist":
                    print("satisfied if updatedtype == whitelist:")
                    if memberId in faceData['blacklist_ids']:
                        print('memberid is in blacklist')
                        index = faceData['blacklist_ids'].index(memberId)
                        faceData['whitelist_ids'].append(faceData['blacklist_ids'].pop(index))
                        faceData['whitelist_faces'].append(faceData['blacklist_faces'].pop(index))
                        threading.Thread(target = self.faceLmdbOPSobj.insertUnknown,args = (faceData,)).start()
            logger.info("updated data")
        except Exception as e:
            logger.error("An error while trying to update face", exc_info=e)

    def createPushFaceDataInLMDB(self,mem_data):
        for each_member in mem_data:

            if each_member['updated']:
                threading.Thread(target = self.updateMemberType,args = (each_member,)).start()
            
            if each_member['member'][0]['type'] in ['whitelist','blacklist']:
                # i = i+1
                encodings, class_type, memberId = convertMemData2encoding(each_member)

                if class_type == "whitelist" and encodings is not None:
                    # logger.info("updating whitelist")
                    faceData['whitelist_faces'].append(encodings)
                    faceData['whitelist_ids'].append(memberId)

                if class_type == "blacklist" and encodings is not None:
                    # logger.info("updating blacklist")
                    faceData['blacklist_faces'].append(encodings)
                    faceData['blacklist_ids'].append(memberId)

                # if class_type == "unknown" and encodings is not None:
                #     faceData['unknown_faces'].append(encodings)
                #     faceData['unknown_ids'].append(memberId)
        return faceData

    def load_lmdb_fst(self,mem_data_queue):
        while True:
            try:
                logger.info("Starting to listen to members queue")
                mem_data = mem_data_queue.get()
                logger.info("got data from queue")
                with db_env.begin(db=faceDataDictDB) as db_txn:
                    faceData = self.faceLmdbOPSobj.fetchLMDB(db_txn, "faceData")

                logger.info("got face data")
                logger.info(faceData)

                if faceData is None:
                    faceData = {"whitelist_faces":[],"whitelist_ids":[],"blacklist_faces":[],"blacklist_ids":[],"unknown_faces":[],"unknown_ids":[]}

                if len(mem_data) == 1 and mem_data[0]['updated']:
                    logger.info("got member data for updation")
                    threading.Thread(target = self.updateMemberType,args = (mem_data[0],)).start()
                else:
                    logger.info("got member data for insertion")
                    faceData = self.createPushFaceDataInLMDB(mem_data)                
                    threading.Thread(target = self.faceLmdbOPSobj.insertUnknown,args = (faceData,)).start()

            except Exception as e:
                logger.error("An error occurred during inserting face data into LMDB", exc_info=e)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
# def facedetection(im):
#     results = model.predict(im)
#     coords = []
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             b = box.xyxy[0].cpu()
#             c = tuple(b.numpy())
#             d = []
#             for i in c:
#                 i = int(i)
#                 d.append(i)
#             first_element = d.pop(0)
#             # # print(first_element)
#             d.append(first_element)
#             coords.append(tuple(d))
#     return coords



# if tracktype '100'(unidentified) in faceDecisionDictionary with other categories 
# which means the person face got detected in some frames so we can excluse unidentified results and consider identified results to find the category of that person

class FaceRecognition:
    def __init__(self):
        self.faceLmdbOPSobj = lmdboperationsFaceRecognition()


    def most_common_element(self,input_list):
        #creating a list of dids
        listOfMIDs = [list(each.keys())[0] for each in input_list]
        if not listOfMIDs:
            return None

        max_count = 0
        most_common = None

        # selecting a MID from a list of MIDs with maximum occurance
        for eachMID in listOfMIDs:
            count = listOfMIDs.count(eachMID)
            if count > max_count:
                max_count = count
                most_common = eachMID

        # returning the MID
        for each in input_list:
            if list(each.keys())[0] == most_common:
                return each

    
    def faceDecisionmaker(self,faceDecisionDictionary, trackType_DidEncodingsDict):
        max_key = max(faceDecisionDictionary, key=faceDecisionDictionary.get)
        did_dict = self.most_common_element(trackType_DidEncodingsDict[max_key])
        return did_dict , max_key





    def removeUnidentified(self, faceDecisionDictionary, trackType_DidEncodingsDict):
        if len(faceDecisionDictionary) > 1 and '100' in faceDecisionDictionary:
            removed_value = faceDecisionDictionary.pop('100')
            trackType_DidEncodingsDict.pop('100')
        logger.info(faceDecisionDictionary)
        return faceDecisionDictionary



    def identifyUnknownAndUpdateLMDB(self, encodings, track_type, did):
        if encodings is not None and track_type == '10':
            try:
                with db_env.begin(db=faceDataDictDB) as db_txn:
                    faceData = self.faceLmdbOPSobj.fetchLMDB(db_txn, "faceData")
                encodings = encodings[0]
        
                faceData['unknown_faces'].append(encodings)
        
                faceData['unknown_ids'].append(did)
        
                threading.Thread(target = self.faceLmdbOPSobj.insertUnknown,args = (faceData,)).start()
        
            except Exception as e:
                print("error", e)

    def loadVarsForFaceRecog(self, im0, faceData):
        did = ""
        track_type = "100"
        print(len(faceData['whitelist_faces']),len(faceData['blacklist_faces']),len(faceData['unknown_faces']))
        print(len(faceData['whitelist_ids']),len(faceData['blacklist_ids']),len(faceData['unknown_ids']))
        whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids = faceData['whitelist_faces'],faceData['blacklist_faces'],faceData['unknown_faces'],faceData['whitelist_ids'],faceData['blacklist_ids'],faceData['unknown_ids']
        
        minimum_distance = []

        image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        locations = face_recognition.face_locations(image)
        crop = None
        for location in locations:
            y1, x2, y2, x1 = location
            crop = image[y1:y2, x1:x2]
        return did, track_type, whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids, minimum_distance, crop, image, locations
        

    def faceRecognition(self, im0):
        # sourcery skip: extract-duplicate-method, inline-variable, low-code-quality, remove-unnecessary-else, swap-if-else-branches
        
        with db_env.begin(db=faceDataDictDB) as db_txn:
            faceData = self.faceLmdbOPSobj.fetchLMDB(db_txn, "faceData")
        # print("inside faceRecognition(im0)")
        # print(type(faceData))
        if faceData:
            did, track_type, whitelist_faces, blacklist_faces, unknown_faces, whitelist_ids, blacklist_ids, unknown_ids, minimum_distance, crop, image, locations = self.loadVarsForFaceRecog(im0, faceData)
            if len(locations) != 0:

                randd = randomword()
                encodings = face_recognition.face_encodings(image,locations)
                if len(whitelist_faces):
                    for face_encoding ,face_location in zip(encodings, locations):
                        matches_white = face_recognition.compare_faces(whitelist_faces,face_encoding)
                        faceids_white = face_recognition.face_distance(whitelist_faces,face_encoding)
                        matchindex_white = np.argmin(faceids_white)
                        if min(faceids_white) <=0.70 and matches_white[matchindex_white]:
                            did = str(whitelist_ids[matchindex_white])
                            track_type = "00"
                            return [did, track_type, encodings, crop]

                if len(blacklist_faces):
                    for face_encoding ,face_location in zip(encodings, locations):
                        matches_black = face_recognition.compare_faces(blacklist_faces,face_encoding)
                        faceids_black = face_recognition.face_distance(blacklist_faces,face_encoding)
                        matchindex_black = np.argmin(faceids_black)
                        if min(faceids_black) <=0.70 and matches_black[matchindex_black]:
                            did = str(blacklist_ids[matchindex_black])
                            track_type = "01"
                            return [did, track_type, encodings, crop]
                if len(unknown_faces):
                    for face_encoding ,face_location in zip(encodings, locations):
                        matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                        faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                        matchindex_unknown = np.argmin(faceids_unknown)
                        minimum_distance.append(min(faceids_unknown))
                        if min(faceids_unknown) <=0.70:
                            did = str(unknown_ids[matchindex_unknown])
                            track_type = "11"
                            return [did, track_type, encodings, crop]
                        else:
                            did = str(uuid.uuid4())
                            track_type = "10"
                            if id not in unknown_ids:
                                return [did, track_type, encodings, crop]
                else:
                    did = str(uuid.uuid4())
                    track_type = "10"
                    if id not in unknown_ids:
                        return [did, track_type, encodings, crop]
            else:
                did = ""
                track_type = "100"
                return [did, track_type, im0, im0]
        else:
            did = ""
            track_type = "100"
            return [did, track_type, im0, im0]


    def find_person_type(self, listOfCrops):
        # global unknown_faces, unknown_ids

        faceDecisionDictionary = {}
        trackType_DidEncodingsDict = {}

        #listOfCrops - list of selected body crops for a person
        for oneCrop in listOfCrops:

            try:
                did, track_type, encodings, crop = self.faceRecognition(oneCrop[0])

                #for efficient decision making    
                # faceDecisionDictionary = {'tracktype':No of times facerecog model detected the person as this tracktype} 00:9, 01:3, 100:2,10:6
                if track_type not in faceDecisionDictionary:
                    faceDecisionDictionary[track_type] = 1
                faceDecisionDictionary[track_type] = faceDecisionDictionary[track_type] + 1            

                #creating a dictionary with 'track type' as key and list of {'did':'encodings'} as value     
                #trackType_DidEncodingsDict = {'track type':[{'did':'encodings'},{'did':'encodings'}...]}
                if track_type not in trackType_DidEncodingsDict:
                    trackType_DidEncodingsDict[track_type] = []
                trackType_DidEncodingsDict[track_type].append({did:encodings})
                
            except Exception as e:
                logger.error("An error occurred while converting path to cid", exc_info=e)
                
        # TODO: check weather this condition is needed or not
        # if '' in faceDecisionDictionary:
        #     removed_value = faceDecisionDictionary.pop('')
        #     trackType_DidEncodingsDict.pop('')

        faceDecisionDictionary = self.removeUnidentified(faceDecisionDictionary, trackType_DidEncodingsDict)
        
        # uses {'tracktype':No of times facerecog model detected the person as this tracktype} and {'track type':[{'did':'encodings'},{'did':'encodings'}...]}
        # finds a category based on most common track type among all the detections
        # then in trackType_DidEncodingsDict fetched the [{'did':'encodings'},{'did':'encodings'}...] for that tracktype
        did_dict, track_type = self.faceDecisionmaker(faceDecisionDictionary, trackType_DidEncodingsDict)

        for key, value in did_dict.items():
            did = key
            encodings = value

        #at this point, we have memberid, tracktype and encoding of that person
        #update the LMDB if the person is identifies as "unknown"
        self.identifyUnknownAndUpdateLMDB(encodings, track_type, did)

        return track_type+did, track_type, crop


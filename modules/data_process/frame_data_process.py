from modules.components.load_paths import *
from init import loadLogger

import torch
from pathlib import Path
import os
import cv2
import uuid
import copy
import subprocess as sp
from os.path import join, dirname
from dotenv import load_dotenv
import ast
import nats, json
import numpy as np
import asyncio
import lmdb
import threading
from torch.multiprocessing import Process, set_start_method
from modules.deepstream.rtsp2rtsp import framedata_queue
from collections import deque

from modules.alarm.alarm_light_trigger import alarm
from modules.components.batchdata2json import output_func
from modules.db.db_insert import dbpush_activities, dbpush_members
from modules.face_recognition_pack.recog_objcrop_face import FaceRecognition
from modules.mini_mmaction.demo.demo_spatiotemporal_det import activity_main
from modules.lmdbSubmodules.liveStreamLabelGen import lmdboperations
from modules.data_process.labelTitle import fetch_batch_title

logger = loadLogger()
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")
alarm_config = os.getenv("alarm_config")

anamoly_object = ast.literal_eval(os.getenv("anamoly_object"))
anamoly = ast.literal_eval(os.getenv("anamoly"))
anamolyMemberCategory = ast.literal_eval(os.getenv('anamolyMemberCategory'))
batch_size = int(os.getenv("batch_size"))

db_env = lmdb.open(f'{lmdb_path}/face-detection', max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

batch = []
frame_cnt = 0
isolate_queue = {}

#TODO: 
#move to .env 
trigger_age = 50

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def conv_path2cid(pathh):
    try:
        # logger.info("trying to convert path to cid in json")
        command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(file_path=pathh,ipfs_url=ipfs_url)
        output = sp.getoutput(command)
        logger.info("converted path to cid in json")
        return output
    
    except Exception as e:
        logger.error("An error occurred while converting path to cid", exc_info=e)

def conv_jsonnumpy_2_jsoncid(primary):
    try:
        # logger.info("trying to convert numpy to cid in json")
        if not os.path.exists(ipfs_tempdata_path):
            os.makedirs(ipfs_tempdata_path)
        if not os.path.exists(f"{ipfs_tempdata_path}/" + primary['deviceid']):
            os.makedirs(f"{ipfs_tempdata_path}/" + primary['deviceid'])

        pathh = f"{ipfs_tempdata_path}/" + primary['deviceid'] + "/cid_ref_full.jpg"
        cv2.imwrite(pathh,primary["metaData"]['cid'][0])
        primary["metaData"]['cid'] = conv_path2cid(pathh)

        for each in primary["metaData"]["object"]:
            pathh = f"{ipfs_tempdata_path}/" + primary['deviceid'] + "/cid_ref.jpg"



            cv2.imwrite(pathh,np.array(each["cids"][0], dtype=np.uint8))
            each["cids"] = conv_path2cid(pathh)
        logger.info("converted numpy to cid in json")

        return primary
    except Exception as e:
        logger.info("<<<<<<<<<<<<<<<None type error checking>>>>>>>>>>>>>>>")

        logger.info(pathh)
        logger.info(each["cids"])
        # logger.info(type(each["cids"][0]))
        logger.info(type(np.array(each["cids"][0])))

        logger.info("<<<<<<<<<<<<<<<None type error checking>>>>>>>>>>>>>>>")
        logger.error("An error occurred while converting numpy to cid in a output json", exc_info=e)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def updateData(data_dict, new_member_id):
    try:
        logger.info("trying to update face by removing duplicate track ids")
        keys_to_remove = [
            key
            for key, value in data_dict.items()
            if 'memberID' in value and value['memberID'] == new_member_id
        ]
        for key in keys_to_remove:
            del data_dict[key]

        logger.info("updated face by removing duplicate track ids")
        return data_dict,keys_to_remove
    except Exception as e:
        logger.error("An error occurred during updating face by removing duplicate track ids", exc_info=e)

def updateLMDBAndDataVar(listOfCrops, detection):
    try:
        FaceRecognitionObj = FaceRecognition()
        lmdboperationsobj = lmdboperations()
        with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
            data = lmdboperationsobj.fetchLMDB(db_txn, "trackIdMemIdDict")
        #TODO: need to handle temp fix
        encodings = None
        if data is None:
            did, track_type, encodings = FaceRecognitionObj.find_person_type(listOfCrops)
            data = {}
            if len(detection['cropsNumpyList'])>20:
                data[detection['id']] = [did, track_type]
                # logger.info("insertLMDB error debug")
                with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:

                    lmdboperationsobj.insertLMDB(db_txn, "trackIdMemIdDict", data)
        elif detection['id'] in data:
            #TODO:fetch encodings from face data in LMDB|there is error encodings referenced before assignment in line detection['cids'] = encodings
            did, track_type = data[detection['id']]
        else:
            did, track_type, encodings = FaceRecognitionObj.find_person_type(listOfCrops)
            if len(detection['cropsNumpyList'])>20:
                data[detection['id']] = [did, track_type]
                # logger.info("insertLMDB error debug")
                with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:

                    lmdboperationsobj.insertLMDB(db_txn, "trackIdMemIdDict", data)
        if did == "":
            did = None
        return did, track_type, encodings
    
    except Exception as e:
        logger.error("An error occurred during finding facial results and mapping it with track IDS", exc_info=e)

def updateObjectMetaData(output_json):
    try:
        for detection in output_json['metaData']['object']:
            listOfCrops = detection['cropsNumpyList']

            did, track_type, crops = updateLMDBAndDataVar(listOfCrops, detection)

            detection["track"] = track_type
            detection['memDID'] = str(did)
            # if track_type in ["10","11"]:
            #     detection['cids'] = crops
            del detection['cropsNumpyList']
        return output_json
    except Exception as e:
        logger.error("An error occurred during updating face by removing duplicate track ids and output json meta", exc_info=e)

def mapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res):
    try:
        if trackIdMemIdDict[objId][0] != '100':
            data,keys_to_remove = updateData(data, trackIdMemIdDict[objId][0])
            idsToBeRemoved.extend(iter(keys_to_remove))
            data[objId]["memberID"] = trackIdMemIdDict[objId][0]
        if data[objId]["memberID"] is None:
            data[objId]["memberID"] = trackIdMemIdDict[objId][0]
        if len(data[objId]["activity"]) >= 4:
            data[objId]["activity"].pop(0)
        data[objId]["activity"].append(act_batch_res[int(objId)])
        return data
    except Exception as e:
        logger.error("An error occurred during mapping Object ID With Activity", exc_info=e)

def createAndmapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res):
    try:
        actList = [act_batch_res[int(objId)]]
        if trackIdMemIdDict[objId][0] != '100':
            data,keys_to_remove = updateData(data, trackIdMemIdDict[objId][0])
            idsToBeRemoved.extend(iter(keys_to_remove))
        data[objId] = {"memberID":trackIdMemIdDict[objId][0],"activity":actList}
        return data
    except Exception as e:
        logger.error("An error occurred during creating and mapping Object ID With Activity", exc_info=e)

def mapIdActivityInLabelinfo(act_batch_res, trackIdMemIdDict, idsToBeRemoved):
    try:
        lmdboperationsobj = lmdboperations()
        for objId in act_batch_res:
            objId = str(objId)

            if objId in trackIdMemIdDict:
                with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                    data = lmdboperationsobj.fetchLMDB(db_txn, "IdLabelInfo")
                if data is None:
                    data = {}
                elif objId in data:
                    data = mapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res)
                else:
                    data = createAndmapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res)
                # logger.info("insertLMDB error debug")
                lmdboperationsobj.insertLMDB(db_txn, "IdLabelInfo", data)
                logger.info("updateddata after face recognition", data)
                logger.info("\n")
        return idsToBeRemoved
    except Exception as e:
        logger.error("An error occurred during updating LMDB (mapping Object ID With Activity)", exc_info=e)

def face_recognition_process(output_json, device_id, act_batch_res):
    try:
        lmdboperationsobj = lmdboperations()
        idsToBeRemoved = []
        logger.info("started face recognition")

        output_json = updateObjectMetaData(output_json)

        with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
            trackIdMemIdDict = lmdboperationsobj.fetchLMDB(db_txn, "trackIdMemIdDict")
        idsToBeRemoved = mapIdActivityInLabelinfo(act_batch_res, trackIdMemIdDict, idsToBeRemoved)

        logger.info("idsToBeRemoved")
        logger.info(idsToBeRemoved)

        output_json = conv_jsonnumpy_2_jsoncid(output_json)



        output_json['metaData']['object'] = [
            obj for obj in output_json['metaData']['object'] if obj['id'] not in idsToBeRemoved
        ]
        
        output_json = fetch_batch_title(output_json)
        print(output_json['metaData']['title'])

        logger.info("\n")
        logger.info(f'updated FACE OUTPUT: {output_json}')

        dbpush_members(output_json)

    except Exception as e:
        logger.info("<<<<<<<<<<<<<<<None type error checking>>>>>>>>>>>>>>>")
        logger.info(output_json)
        for obj in output_json['metaData']['object']:
            if obj['id'] not in idsToBeRemoved:
                logger.info(obj)
        logger.info("<<<<<<<<<<<<<<<None type error checking>>>>>>>>>>>>>>>")
        logger.error("An error occurred during facial recognition", exc_info=e)        

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
async def json_publish_activity(primary):
    try:
        logger.info('starting to publish anomalies through nats')
        nc = await nats.connect(servers=nats_urls , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
        js = nc.jetstream()
        JSONEncoder = json.dumps(primary)
        json_encoded = JSONEncoder.encode()
        Subject = "service.notifications"
        ack = await js.publish(Subject, json_encoded)
        logger.info(f'Ack: stream={ack.stream}, sequence={ack.seq}')
        logger.info("Activity is getting published")
    except Exception as e:
        logger.error("An error occurred while publishing the anomaly", exc_info=e)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def merge_activity(act_batch_res, output_json):
    try:
        logger.info('starting merge_activity function')
        if len(output_json['metaData']['object'])>0:
            for obj in output_json['metaData']['object']:
                if int(obj["id"]) in act_batch_res:
                    obj["activity"] = act_batch_res[int(obj["id"])]
        return output_json
    except Exception as e:
        logger.error("An error occurred merge_activity function", exc_info=e)

def updateLabelInfo(data, act_batch_res,objId):
    try:
        if data is None:
            actList = [act_batch_res[int(objId)]]
            data = {objId: {"memberID": None, "activity": actList}}
        elif objId in data:
            if len(data[objId]["activity"]) >= 4:
                data[objId]["activity"].pop(0)
            data[objId]["activity"].append(act_batch_res[int(objId)])
        else:
            actList = [act_batch_res[int(objId)]]
            data[objId] = {"memberID":None,"activity":actList}
        return data
    except Exception as e:
        logger.error("An error occurred while updating label info in LMDB", exc_info=e)

def updateOutputJsonWithActivity(org_frames_lst,bbox_tensor_lst,obj_id_ref, subscriptions, output_json):
    act_batch_res={}
    try:
        lmdboperationsobj = lmdboperations()
        if 'Activity' in subscriptions:
            act_batch_res = activity_main(org_frames_lst,bbox_tensor_lst,obj_id_ref)
            for objId in act_batch_res:
                objId = str(objId)
                
                with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                    data = lmdboperationsobj.fetchLMDB(db_txn, "IdLabelInfo")
                    data = updateLabelInfo(data, act_batch_res,objId)
                    lmdboperationsobj.insertLMDB(db_txn, "IdLabelInfo", data)
            output_json = merge_activity(act_batch_res, output_json)
        return output_json, act_batch_res
    except Exception as e:
        logger.error("An error occurred while updating output json with activity", exc_info=e)

def updateOutputJsonWithMetadatas(output_json, device_data, device_id, device_id_new, device_timestamp, fin_full_frame):
    try:
        batchId = str(uuid.uuid4())
        output_json["tenantId"] = device_data[device_id]['tenantId']
        output_json["batchid"] = batchId
        output_json["deviceid"] = device_id_new
        output_json["timestamp"] = str(device_timestamp)
        output_json['geo']['latitude'] = device_data[device_id]['lat']
        output_json['geo']['longitude'] = device_data[device_id]['long']
        output_json["metaData"]['detect'] = len(output_json["metaData"]['object'])
        output_json["metaData"]['count']["peopleCount"] = len(output_json["metaData"]['object'])
        output_json["version"] = "v0.0.4"
        if fin_full_frame is not None:
            output_json["metaData"]["cid"] = fin_full_frame
        return output_json
    except Exception as e:
        logger.error("An error occurred while updating output json with metadatas", exc_info=e)

def checkForActivityAlarm(subscriptions):
    try:
        if "Alarm" in subscriptions:
            logger.info("alarm trigger for activities")
            alarm()
    except Exception as e:
        logger.error("alarm is not connected / couldn't connect to alarm", exc_info=e) 

def processOutputjsonForDBPush(output_json):
    try:
        output_json = fetch_batch_title(output_json)
        output_json_fr = copy.deepcopy(output_json)

        output_json = conv_jsonnumpy_2_jsoncid(output_json)

        for detection in output_json['metaData']['object']:
            del detection['cropsNumpyList']   
        return output_json,output_json_fr
    except Exception as e:
        logger.error("An error occurred while processing output json", exc_info=e)

def checkForAnomaly(output_json):
    #TODO:check the condition
    try:
        return(
        [
            True
            for each in [
                each["activity"]
                for each in output_json['metaData']['object']
            ]
            if each in anamoly
        ]
        or [
            True
            for each in [
                each["class"]
                for each in output_json['metaData']['object']
            ]
            if each in anamoly_object
        ]
        or [
            True
            for each in [
                each["track"]
                for each in output_json['metaData']['object']
            ]
            if each in anamolyMemberCategory
        ])
    except Exception as e:
        logger.error("An error occurred while creating flag for anomaly", exc_info=e)

def publishPushProcessOutputJson(subscriptions, output_json, AnomalyFlag, output_json_fr,device_id, act_batch_res):
    try:
        #TODO:handle else case
        if 'Bagdogra' not in subscriptions:
            status = dbpush_activities(output_json)
            if(status == "SUCCESS!!"):
                if AnomalyFlag:
                    asyncio.run(json_publish_activity(primary=output_json))
                logger.info("DB insertion successful :)")
                if 'Facial-Recognition' in subscriptions:
                    face_recognition_process(output_json_fr,device_id, act_batch_res)
                    # asyncio.create_task(face_recognition_process(output_json_fr,device_id))
                    # threading.Thread(target = face_recognition_process,args = (output_json_fr,device_id,)).start()
            elif(status == "FAILURE!!"):
                logger.info("DB insertion got failed :(")
        
    except Exception as e:
        logger.error("An error occurred while publishing/pushing the output json", exc_info=e)

def anomalyTagAndTrigger(subscriptions, output_json):
    try:
        if AnomalyFlag := checkForAnomaly(output_json):
            threading.Thread(target = checkForActivityAlarm,args = (subscriptions,)).start()
            output_json["type"] = "anomaly"
        return output_json, AnomalyFlag
    except Exception as e:
        logger.error("An error occurred while anomaly tag and trigger", exc_info=e)

def process_results(device_id, device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame):
    try:
        logger.info("entering process results")
        subscriptions = device_data[device_id]["subscriptions"]
        output_json, act_batch_res = updateOutputJsonWithActivity(org_frames_lst,bbox_tensor_lst,obj_id_ref, subscriptions, output_json)

        if output_json['metaData']['object']:
            output_json = updateOutputJsonWithMetadatas(output_json, device_data, device_id, device_id_new, device_timestamp, fin_full_frame)
            output_json, AnomalyFlag = anomalyTagAndTrigger(subscriptions, output_json)
            output_json,output_json_fr = processOutputjsonForDBPush(output_json)
            logger.info("THE OUTPUT JSON STRUCTURE: ",output_json)
            publishPushProcessOutputJson(subscriptions, output_json, AnomalyFlag, output_json_fr,device_id, act_batch_res)

    except Exception as e:
        logger.error("error in processing the results", exc_info=e)
        
def process_publish(device_id,batch_data,device_data,device_timestamp):
    try:
        logger.info(f"got batch data for device ID:+{str(device_id)}")
        
        global anamoly_object, anamoly
        device_id_new = device_data[device_id]["deviceId"]
        detectss = 0
        fin_full_frame = None

        for each in batch_data:
            if each["total_detects"] > detectss:
                detectss = each["total_detects"]
                fin_full_frame = each["det_frame"]
        
        bbox_tensor_lst = [frame_data["bbox_tensor"] for frame_data in batch_data]
        org_frames_lst = [frame_data["org_frame"] for frame_data in batch_data]
        obj_id_ref = [frame_data["bbox_ref_list"] for frame_data in batch_data]

        try:
            output_json = output_func(batch_data,device_id,device_timestamp)
        except Exception as e:
            logger.error("An error occurred outside function output_func", exc_info=e)
        
        try:
            threading.Thread(target = process_results, args = (device_id, device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame,)).start()
        except Exception as e:
            logger.error("An error occurred outside function process_results", exc_info=e)
            
    except Exception as e:
        logger.error("An error occurred during structuring output json", exc_info=e)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def loadVariableForFrame2Dict(inputt,dev_id_dict):
    try:
        global anamoly_object, anamoly
        global trigger_age
        device_id = inputt["camera_id"]
        subscriptions = dev_id_dict[device_id]["subscriptions"]
        frame_timestamp = inputt["frame_timestamp"]
        frame_data = []
        bbox_list = []
        bbox_ref_list = []
        
        return anamoly_object, trigger_age, device_id, subscriptions, frame_timestamp, frame_data, bbox_list, bbox_ref_list
    except Exception as e:
        logger.error("An error occurred while loading variable for frametodict", exc_info=e)

def checkForAnomalousObject(objectt, anamoly_object, trigger_age):
    try:
        return (
            objectt["detect_type"] in anamoly_object
            and objectt["age"] > trigger_age
        )
    except Exception as e:
        logger.error("An error occurred while checking for anomalous object to trigger alarm", exc_info=e)

def checkForobjectAlarm(objectt, anamoly_object, trigger_age, subscriptions):
    try:
        if "Alarm" in subscriptions and checkForAnomalousObject(objectt, anamoly_object, trigger_age):
            trigger_age = trigger_age + 1
            print(f"Alarm triggered for {objectt['detect_type']} age: {str(objectt['age'])}")
            try:
                alarm()
            except Exception:
                print("alarm is not connected / couldn't connect to alarm")
    except Exception as e:
        logger.error("An error occurred while checking for object alarm", exc_info=e)

def structureobjdict(object_data):
    try:
        return {
                object_data["obj_id"]: {
                    "type": object_data["detect_type"],
                    "activity": "No Activity",
                    "confidence": object_data["confidence_score"],
                    "did": None,
                    "track_type": None,
                    "crops": [object_data["crop"]]
                }
            }
    except Exception as e:
        logger.error("An error occurred while structing object dict for frametodict", exc_info=e)

def createBBocCoord(object_data, bbox_ref_list, bbox_list):
    try:
        bbox_coord = [object_data["bbox_left"], object_data["bbox_top"], object_data["bbox_right"], object_data["bbox_bottom"]]
        bbox_ref_list.append(object_data["obj_id"])
        bbox_list.append(bbox_coord)
        return bbox_ref_list, bbox_list
    except Exception as e:
        logger.error("An error occurred while creating bbox coord for frametodict", exc_info=e)

def createObjectsDict(inputt, subscriptions, frame_data, bbox_ref_list, bbox_list, trigger_age, anamoly_object):
    try:
        if len(inputt["objects"]) > 0:
            for object_data in inputt["objects"]:
                # #object_data['crop']
                # reid = func(object_data['crop'])
                threading.Thread(target = checkForobjectAlarm,args = (object_data, anamoly_object, trigger_age, subscriptions,)).start()
                obj_dict = structureobjdict(object_data)

                frame_data.append(obj_dict)

                if object_data["detect_type"] in ["Male", "Female"]:
                    bbox_ref_list, bbox_list = createBBocCoord(object_data, bbox_ref_list, bbox_list)
        return frame_data, bbox_ref_list, bbox_list
    except Exception as e:
        logger.error("An error occurred while creating objects dict for frametodict", exc_info=e)

def batchingFrames(device_id, finalFrameDict, dev_id_dict, frame_timestamp):
    try:
        if device_id not in isolate_queue:
            isolate_queue[device_id] = []
        isolate_queue[device_id].append(finalFrameDict)
        for each in isolate_queue:
            if len(isolate_queue[each])>batch_size:
                batch_data = isolate_queue[each]
                isolate_queue[each] = []
                #TODO:put in queue
                logger.info("sending a batch of frames to process")
                process_publish(device_id,batch_data,dev_id_dict, frame_timestamp)
                # asyncio.create_task(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp))
    except Exception as e:
        logger.error("An error occurred while batching frames", exc_info=e)

def structureFinalFrameDict(inputt,frame_data, cidd, bbox_array, bbox_ref_list):
    try:
        return{
            "frame_id":inputt["frame_number"],
        "detection_info":frame_data,
        "cid":cidd, 
        "bbox_tensor":bbox_array, 
        "org_frame":inputt["org_frame"], 
        "bbox_ref_list":bbox_ref_list, 
        "total_detects":inputt["total_detect"], 
        "det_frame":cidd
        }
    except Exception as e:
        logger.error("An error occurred while structing final frame dict for frametodict", exc_info=e)

def frame_2_dict():
    while True:
        try:
            inputt, dev_id_dict = framedata_queue.get()
            # print(inputt,"Main inputt dict")
            
            anamoly_object, trigger_age, device_id, subscriptions, frame_timestamp, frame_data, bbox_list, bbox_ref_list = loadVariableForFrame2Dict(inputt,dev_id_dict)

            frame_data, bbox_ref_list, bbox_list = createObjectsDict(inputt, subscriptions, frame_data, bbox_ref_list, bbox_list, trigger_age, anamoly_object)
            bbox_np = np.array(bbox_list, dtype=np.float32)
            bbox_array = np.reshape(bbox_np, (-1, 4))
            cidd = [inputt["np_arr"]]
            finalFrameDict = structureFinalFrameDict(inputt,frame_data, cidd, bbox_array, bbox_ref_list)
            batchingFrames(device_id, finalFrameDict, dev_id_dict, frame_timestamp)

        except Exception as e:
            logger.error("An error occurred outside function createRTSPPort", exc_info=e)



                    









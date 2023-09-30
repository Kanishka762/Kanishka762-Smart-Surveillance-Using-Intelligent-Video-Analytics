from modules.components.load_paths import *
from init import loadLogger
import threading
from dotenv import load_dotenv
import numpy as np
import ast
import lmdb
import uuid
import asyncio
import copy

from modules.lmdbSubmodules.liveStreamLabelGen import lmdboperations
from modules.mini_mmaction.demo.demo_spatiotemporal_det import activity_main
from modules.components.batchdata2json import output_func
from modules.db.db_insert import dbpush_activities
from modules.data_process.labelTitle import fetch_batch_title
from modules.data_process.frame_data_process import conv_jsonnumpy_2_jsoncid, json_publish_activity, face_recognition_process
from modules.alarm.alarm_light_trigger import alarm

logger = loadLogger()
load_dotenv(dotenv_path)

db_env = lmdb.open(f'{lmdb_path}/face-detection', max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

anamolyMemberCategory = ast.literal_eval(os.getenv('anamolyMemberCategory'))
anamoly_object = ast.literal_eval(os.getenv("anamoly_object"))
anamoly = ast.literal_eval(os.getenv("anamoly"))



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
        logger.info("inside updateOutputJsonWithActivity")
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
            logger.info("THE OUTPUT JSON STRUCTURE: ")
            logger.info(output_json)
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

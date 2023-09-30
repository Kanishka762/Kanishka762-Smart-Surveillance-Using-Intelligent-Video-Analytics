from modules.components.load_paths import *
from init import loadLogger

from pathlib import Path
import os
import cv2
import subprocess as sp
# from os.path import join, dirname
from dotenv import load_dotenv
import ast
import nats, json
import numpy as np
import lmdb

# from collections import deque

from modules.db.db_insert import dbpush_members
from modules.face_recognition_pack.recog_objcrop_face import FaceRecognition
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
batch_size = int(os.getenv("batch_size"))

db_env = lmdb.open(f'{lmdb_path}/face-detection', max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

batch = []
frame_cnt = 0

#TODO: 
#move to .env 
trigger_age = 50

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TODO:convert into class
def conv_path2cid(pathh):
    try:
        # logger.info("trying to convert path to cid in json")
        command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(file_path=pathh,ipfs_url=ipfs_url)
        output = sp.getoutput(command)
        logger.info("converted path to cid in json")
        return output
    
    except Exception as e:
        logger.error("An error occurred while converting path to cid", exc_info=e)

#TODO:convert into class
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
        logger.info("inside updateLMDBAndDataVar")

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

def mapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res, data):
    try:
        logger.info(trackIdMemIdDict)
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

def createAndmapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res, data):
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
        logger.info("inside mapIdActivityInLabelinfo")
        
        lmdboperationsobj = lmdboperations()
        for objId in act_batch_res:
            objId = str(objId)

            if objId in trackIdMemIdDict:
                with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                    data = lmdboperationsobj.fetchLMDB(db_txn, "IdLabelInfo")
                    if data is None:
                        data = {}
                    elif objId in data:
                        data = mapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res, data)
                    else:
                        data = createAndmapObjIDWithActivity(trackIdMemIdDict, objId, idsToBeRemoved, act_batch_res, data)
                    # logger.info("insertLMDB error debug")
                    lmdboperationsobj.insertLMDB(db_txn, "IdLabelInfo", data)
                    logger.info("updateddata after face recognition")
                    logger.info(data)
        return idsToBeRemoved
    except Exception as e:
        logger.error("An error occurred during updating LMDB (mapping Object ID With Activity)", exc_info=e)

def face_recognition_process(output_json, device_id, act_batch_res):
    try:
        logger.info("inside face_recognition_process")
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
#TODO:convert into class
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


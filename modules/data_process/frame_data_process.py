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
from modules.deepstream.rtsp2frames import framedata_queue
from collections import deque

from modules.alarm.alarm_light_trigger import alarm
from modules.components.batchdata2json import output_func
from modules.db.db_insert import dbpush_activities, dbpush_members
from modules.face_recognition_pack.recog_objcrop_face import find_person_type
from modules.mini_mmaction.demo.demo_spatiotemporal_det import activity_main

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

db_env = lmdb.open(lmdb_path+'/face-detection',
                max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

batch = []
frame_cnt = 0
isolate_queue = {}

#TODO: 
#move to .env 
trigger_age = 50


def conv_path2cid(pathh):
    try:
        logger.info("trying to convert path to cid in json")
        command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(file_path=pathh,ipfs_url=ipfs_url)
        output = sp.getoutput(command)
        logger.info("converted path to cid in json")
        return output
    except:
        logger.error("An error occurred while converting path to cid", exc_info=e)

def conv_jsonnumpy_2_jsoncid(primary):
    try:
        logger.info("trying to convert numpy to cid in json")
        if not os.path.exists(ipfs_tempdata_path):
            os.makedirs(ipfs_tempdata_path)
        if not os.path.exists(ipfs_tempdata_path+"/"+primary['deviceid']):
            os.makedirs(ipfs_tempdata_path+"/"+primary['deviceid'])
        
        pathh = ipfs_tempdata_path+"/"+primary['deviceid']+"/cid_ref_full.jpg"
        cv2.imwrite(pathh,primary["metaData"]['cid'][0])
        primary["metaData"]['cid'] = conv_path2cid(pathh)

        for each in primary["metaData"]["object"]:
            pathh = ipfs_tempdata_path+"/"+primary['deviceid']+"/cid_ref.jpg"
            cv2.imwrite(pathh,np.array(each["cids"][0], dtype=np.uint8))
            each["cids"] = conv_path2cid(pathh)
        logger.info("converted numpy to cid in json")
        
        return primary
    except Exception as e:
        logger.error("An error occurred while converting numpy to cid in a output json", exc_info=e)

def fetchLMDB(db_txn, key):
    try:
        logger.info("trying updating data in LMDB")
        value = db_txn.get(key.encode())
        if value is not None:
            data = json.loads(value.decode())
            logger.info("updated data in LMDB")
            return data
        else:
            return None
    except Exception as e:
        logger.error("An error occurred while updating data in LMDB", exc_info=e)

def insertLMDB(db_txn, key,value):
    try:
        logger.info("trying inserting data in LMDB")
        db_txn.put(key.encode(), json.dumps(value).encode())
        logger.info("inserted data in LMDB")
    except Exception as e:
        logger.error("An error occurred while inserting data in LMDB", exc_info=e)

def updateData(data_dict, new_member_id):
    try:
        logger.info("trying to update face by removing duplicate track ids")
        keys_to_remove = []

        for key, value in data_dict.items():
            if 'memberID' in value and value['memberID'] == new_member_id:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del data_dict[key]

        logger.info("updated face by removing duplicate track ids")
        return data_dict,keys_to_remove
    except Exception as e:
        logger.error("An error occurred during updating face by removing duplicate track ids", exc_info=e)

def face_recognition_process(output_json, device_id, act_batch_res):
    try:
        idsToBeRemoved = []
        logger.info("started face recognition")

        for detection in output_json['metaData']['object']:
            listOfCrops = detection['cropsNumpyList']

            with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
                data = fetchLMDB(db_txn, "trackIdMemIdDict")
                if data is not None:
                    if detection['id'] in data:
                        did, track_type = data[detection['id']]
                    else:
                        did, track_type, encodings = find_person_type(listOfCrops)
                        if len(detection['cropsNumpyList'])>20:
                            data[detection['id']] = [did, track_type]
                            insertLMDB(db_txn, "trackIdMemIdDict", data)
                else:
                    did, track_type, encodings = find_person_type(listOfCrops)
                    data = {}
                    if len(detection['cropsNumpyList'])>20:
                        data[detection['id']] = [did, track_type]
                        insertLMDB(db_txn, "trackIdMemIdDict", data)
            if did == "":
                did = None
            detection["track"] = track_type
            detection['memDID'] = did
            if track_type in ["10","11"]:
                detection['cids'] = encodings
            del detection['cropsNumpyList']

        with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
            trackIdMemIdDict = fetchLMDB(db_txn, "trackIdMemIdDict")
            
        for objId in act_batch_res:
            objId = str(objId)

            if objId in trackIdMemIdDict:
                with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                    data = fetchLMDB(db_txn, "IdLabelInfo")
                    if data is not None:
                        if objId in data:
                            if trackIdMemIdDict[objId][0] != '100':
                                data,keys_to_remove = updateData(data, trackIdMemIdDict[objId][0])
                                for each in keys_to_remove:
                                    idsToBeRemoved.append(each)
                                data[objId]["memberID"] = trackIdMemIdDict[objId][0]
                            if data[objId]["memberID"] is None:
                                data[objId]["memberID"] = trackIdMemIdDict[objId][0]
                            if len(data[objId]["activity"]) < 4:
                                data[objId]["activity"].append(act_batch_res[int(objId)])
                            else:
                                data[objId]["activity"].pop(0)
                                data[objId]["activity"].append(act_batch_res[int(objId)])
                            insertLMDB(db_txn, "IdLabelInfo", data)
                        else:
                            actList = []
                            actList.append(act_batch_res[int(objId)])
                            if trackIdMemIdDict[objId][0] != '100':
                                data,keys_to_remove = updateData(data, trackIdMemIdDict[objId][0])
                                for each in keys_to_remove:
                                    idsToBeRemoved.append(each)
                            data[objId] = {"memberID":trackIdMemIdDict[objId][0],"activity":actList}
                            
                            insertLMDB(db_txn, "IdLabelInfo", data)
                    else:
                        data = {}
                        insertLMDB(db_txn, "IdLabelInfo", data)
                    logger.info("updateddata after face recognition", data)
                    logger.info("\n")

        logger.info("*----------*---------*----------*----------*")

        output_json = conv_jsonnumpy_2_jsoncid(output_json)

        output_json['metaData']['object'] = [
            obj for obj in output_json['metaData']['object'] if obj['id'] not in idsToBeRemoved
        ]

        logger.info("\n")
        logger.info(f'updated FACE OUTPUT: {output_json}')
        with open("./static/test.json", "a") as outfile:
            json.dump(output_json, outfile)
        dbpush_members(output_json)

    except Exception as e:
        logger.error("An error occurred during facial recognition", exc_info=e)

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
        
async def json_publish_activity(primary):
    try:
        logger.info('starting to publish anomalies through nats')
        nc = await nats.connect(servers=nats_urls , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
        js = nc.jetstream()
        JSONEncoder = json.dumps(primary)
        json_encoded = JSONEncoder.encode()
        #TODO:
        #move to .env
        Subject = "service.notifications"
        Stream_name = "services"
        ack = await js.publish(Subject, json_encoded)

        logger.info(f'Ack: stream={ack.stream}, sequence={ack.seq}')
        logger.info("Activity is getting published")
    except Exception as e:
        logger.error("An error occurred while publishing the anomaly", exc_info=e)

def process_results(device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame):
    try:
        logger.info("entering process results")
        subscriptions = device_data[device_id]["subscriptions"]
        act_batch_res={}

        if 'Activity' in subscriptions:
            act_batch_res = activity_main(org_frames_lst,bbox_tensor_lst,obj_id_ref)
            for objId in act_batch_res:
                objId = str(objId)
                with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                    data = fetchLMDB(db_txn, "IdLabelInfo")

                    if data is not None:
                        if objId in data:
                            if len(data[objId]["activity"]) < 4:
                                data[objId]["activity"].append(act_batch_res[int(objId)])
                            else:
                                data[objId]["activity"].pop(0)
                                data[objId]["activity"].append(act_batch_res[int(objId)])
                            insertLMDB(db_txn, "IdLabelInfo", data)
                        else:
                            actList = []
                            actList.append(act_batch_res[int(objId)])
                            data[objId] = {"memberID":None,"activity":actList}
                            insertLMDB(db_txn, "IdLabelInfo", data)
                    else:
                        actList = []
                        actList.append(act_batch_res[int(objId)])
                        data = {}
                        data[objId] = {"memberID":None,"activity":actList}
                        insertLMDB(db_txn, "IdLabelInfo", data)

            output_json = merge_activity(act_batch_res, output_json)

        if output_json['metaData']['object']:
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

            if len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly])>0 or len([True for each in [each["class"] for each in output_json['metaData']['object']] if each in anamoly_object])>0 or len([True for each in [each["track"] for each in output_json['metaData']['object']] if each in anamolyMemberCategory])>0:
                if "Alarm" in subscriptions:
                    logger.info("alarm trigger for activities")
                    try:
                        alarm()
                    except:
                        logger.info("alarm is not connected / couldn't connect to alarm")
                # logger.info("Anomoly check cnt:",len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly]))
                
                output_json["type"] = "anomaly"
                output_json_fr = copy.deepcopy(output_json)
                
                if 'Bagdogra' not in subscriptions:
                    output_json = conv_jsonnumpy_2_jsoncid(output_json)

                for detection in output_json['metaData']['object']:
                    del detection['cropsNumpyList']
                    
                logger.info("THE OUTPUT JSON STRUCTURE: ",output_json)
                
                if 'Bagdogra' not in subscriptions:
                    status = dbpush_activities(output_json)
                    if(status == "SUCCESS!!"):
                        asyncio.run(json_publish_activity(primary=output_json))
                        logger.info("DB insertion successful :)")
                        if 'Facial-Recognition' in subscriptions:
                            face_recognition_process(output_json_fr,device_id, act_batch_res)
                            # asyncio.create_task(face_recognition_process(output_json_fr,device_id))
                            # threading.Thread(target = face_recognition_process,args = (output_json_fr,device_id,)).start()
                    elif(status == "FAILURE!!"):
                        logger.info("DB insertion got failed :(")
            else:
                output_json_fr = copy.deepcopy(output_json)
                output_json = conv_jsonnumpy_2_jsoncid(output_json)
    
                for detection in output_json['metaData']['object']:
                    del detection['cropsNumpyList']
                logger.info("THE OUTPUT JSON STRUCTURE: ",output_json)
                
                if 'Bagdogra' not in subscriptions:
                    status = dbpush_activities(output_json)
                    if(status == "SUCCESS!!"):
                        logger.info("DB insertion successful :)")
                        if 'Facial-Recognition' in subscriptions:
                            face_recognition_process(output_json_fr,device_id, act_batch_res)
                            # asyncio.create_task(face_recognition_process(output_json_fr,device_id))
                            # threading.Thread(target = face_recognition_process,args = (output_json_fr,device_id,)).start()
                        # with open("./static/test.json", "a") as outfile:
                        #     json.dump(output_json, outfile)
                    elif(status == "FAILURE!!"):
                        logger.info("DB insertion got failed :(")   
    except Exception as e:
        logger.error("error in processing the results", exc_info=e)
        
def process_publish(device_id,batch_data,device_data,device_timestamp):
    try:
        logger.info("entering structure output json function")
        
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
            threading.Thread(target = process_results, args = (device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame,)).start()
        except Exception as e:
            logger.error("An error occurred outside function process_results", exc_info=e)
            
    except Exception as e:
        logger.error("An error occurred during structuring output json", exc_info=e)
        
def frame_2_dict():
    while True:
        try:
            inputt, dev_id_dict = framedata_queue.get()
            # logger.info("got data from framedata_queue")
            global anamoly_object, anamoly
            global frame_cnt
            global trigger_age

            device_id = inputt["camera_id"]

            subscriptions = dev_id_dict[device_id]["subscriptions"]

            # face_flag = False

            # for objj in inputt["objects"]:
            #     if objj["detect_type"] == "nomask":
            #         print(inputt)
            #         face_id = str(objj["obj_id"])
            #         face_flag = True
            #         cv2.imwrite("/home/srihari/deepstreambackend/testing_ids/"+str(objj["obj_id"])+"_"+"face.jpg",objj["crop"])
            # if face_flag:
            #     for objj in inputt["objects"]:
            #         if objj["detect_type"] == "Male" or objj["detect_type"] == "Female":
            #             cv2.imwrite("/home/srihari/deepstreambackend/testing_ids/"+str(objj["obj_id"])+"_"+"body.jpg",objj["crop"])


            frame_timestamp = inputt["frame_timestamp"]
            frame_cnt += frame_cnt
            frame_data = []
            bbox_list = []
            bbox_ref_list = []
            if len(inputt["objects"]) > 0:
                for objectt in inputt["objects"]:
                    if "Alarm" in subscriptions and (objectt["detect_type"] in anamoly_object and objectt["age"] > trigger_age):
                        trigger_age = trigger_age + 1

                        print("Alarm triggered for "+objectt["detect_type"]+" age: "+str(objectt["age"]))
                        try:
                            alarm()
                        except Exception:
                            print("alarm is not connected / couldn't connect to alarm")

                    obj_dict = {objectt["obj_id"]: {}}
                    obj_dict[objectt["obj_id"]]["type"] = objectt["detect_type"]
                    obj_dict[objectt["obj_id"]]["activity"] = "No Activity"
                    obj_dict[objectt["obj_id"]]["confidence"] = objectt["confidence_score"]
                    obj_dict[objectt["obj_id"]]["did"] = None
                    obj_dict[objectt["obj_id"]]["track_type"] = None
                    # obj_dict[objectt["obj_id"]]["age"] = objectt["age"]
                    obj_dict[objectt["obj_id"]]["crops"] = [objectt["crop"]]
                    frame_data.append(obj_dict)

                    if objectt["detect_type"] in ["Male", "Female"]:
                        bbox_coord = [objectt["bbox_left"], objectt["bbox_top"], objectt["bbox_right"], objectt["bbox_bottom"]]
                        bbox_ref_list.append(objectt["obj_id"])
                        bbox_list.append(bbox_coord)

            bbox_np = np.array(bbox_list, dtype=np.float32)
            bbox_array = np.reshape(bbox_np, (-1, 4))
            # frame_info_anamoly = anamoly_score_calculator(frame_data)
            # frame_data.clear()
            # frame_anamoly_wgt = frame_weighted_avg(frame_info_anamoly)

            cidd = [inputt["np_arr"]]
            final_frame = {"frame_id":inputt["frame_number"],"detection_info":frame_data,"cid":cidd, "bbox_tensor":bbox_array, "org_frame":inputt["org_frame"], "bbox_ref_list":bbox_ref_list, "total_detects":inputt["total_detect"], "det_frame":cidd}

            if device_id not in isolate_queue:
                isolate_queue[device_id] = []
            isolate_queue[device_id].append(final_frame)
            for each in isolate_queue:
                if len(isolate_queue[each])>batch_size:
                    batch_data = isolate_queue[each]
                    isolate_queue[each] = []
                    process_publish(device_id,batch_data,dev_id_dict, frame_timestamp)
                    # asyncio.create_task(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp))
        except Exception as e:
            logger.error("An error occurred outside function createRTSPPort", exc_info=e)



                    









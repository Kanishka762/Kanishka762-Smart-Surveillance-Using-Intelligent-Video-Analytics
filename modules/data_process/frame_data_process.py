from modules.components.load_paths import *
import torch
from pathlib import Path
from PIL import Image
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
# import pickle
from torch.multiprocessing import Process, set_start_method
from modules.deepstream.rtsp2frames import framedata_queue
from collections import deque
# lock = threading.Lock()
semaphore = threading.Semaphore(1)  # Allow 3 threads at a time
resource = []
# trackIdMemIdDict = {}
# IdLabelInfo = {}
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from modules.alarm.alarm_light_trigger import alarm
from modules.anomaly.anomaly_score import anamoly_score_calculator, frame_weighted_avg
from modules.components.batchdata2json import output_func
from modules.db.db_insert import dbpush_activities, dbpush_members
from modules.face_recognition_pack.recog_objcrop_face import find_person_type
# from warehouse import get_info

import cv2
from modules.mini_mmaction.demo.demo_spatiotemporal_det import activity_main

# path = os.getcwd()
# cwd = os.getcwd()
# data_path = join(cwd, 'data')
# dotenv_path = join(data_path, '.env')
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
# subscriptions = ast.literal_eval(os.getenv("subscriptions"))
batch_size = int(os.getenv("batch_size"))

db_env = lmdb.open(lmdb_path+'/face-detection',
                max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

batch = []
frame_cnt = 0
isolate_queue = {}
trigger_age = 50

print(ipfs_tempdata_path)
def conv_path2cid(pathh):
    command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(file_path=pathh,ipfs_url=ipfs_url)
    output = sp.getoutput(command)
    return output

def conv_jsonnumpy_2_jsoncid(primary):
    # convert full frame numpy to cid
    # print(primary)
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
    # pathh = ipfs_tempdata_path+"/"+primary['deviceid']+"/cid_ref_full.jpg"
    # cv2.imwrite(pathh,np.array(primary["metaData"]['cid'][0], dtype=np.uint8))
    # primary["cid"] = primary["cid"]
    # print(primary)
    return primary

def fetchLMDB(db_txn, key):
    value = db_txn.get(key.encode())
    if value is not None:
        data = json.loads(value.decode())
        return data
    else:
        return None

def insertLMDB(db_txn, key,value):
    db_txn.put(key.encode(), json.dumps(value).encode())

def updateData(data_dict, new_member_id):
    # print("trying to remove ",new_member_id," from ", data_dict) 
    # print("\n")

    keys_to_remove = []
    for key, value in data_dict.items():
        if 'memberID' in value and value['memberID'] == new_member_id:
            keys_to_remove.append(key)
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print(keys_to_remove)
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++")


    for key in keys_to_remove:
        del data_dict[key]

    # print('after removing', data_dict)
    # print("\n") 

    return data_dict,keys_to_remove

def face_recognition_process(output_json, device_id, act_batch_res):
    idsToBeRemoved = []
    # global trackIdMemIdDict
    print("started face recognition")

    for detection in output_json['metaData']['object']:
        listOfCrops = detection['cropsNumpyList']
        # print(len(listOfCrops))
        # print(listOfCrops)

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

    # print("starting")
    # print("\n")

    with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
        trackIdMemIdDict = fetchLMDB(db_txn, "trackIdMemIdDict")
        
    for objId in act_batch_res:
        # print("number of ids and its activity", act_batch_res)
        objId = str(objId)
        # print("objId",objId)
        # print(trackIdMemIdDict)

        if objId in trackIdMemIdDict:
            with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                data = fetchLMDB(db_txn, "IdLabelInfo")
                # print("data from LMDB", data)
                # print("\n")

                # insertLMDBStatus = insertLMDB(db_txn,"IdLabelInfo",data)
                if data is not None:
                    # print("LMDB data: ",data)
                    #update the dict
                    if objId in data:
                        if trackIdMemIdDict[objId][0] != '100':
                            # print('sending for updation',data, trackIdMemIdDict[objId][0])
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
                        #insert the data
                        # print("entered else")
                        actList = []
                        actList.append(act_batch_res[int(objId)])
                        if trackIdMemIdDict[objId][0] != '100':
                            # print('sending for updation',data, trackIdMemIdDict[objId][0])
                            data,keys_to_remove = updateData(data, trackIdMemIdDict[objId][0])
                            for each in keys_to_remove:
                                idsToBeRemoved.append(each)
                        data[objId] = {"memberID":trackIdMemIdDict[objId][0],"activity":actList}
                        
                        insertLMDB(db_txn, "IdLabelInfo", data)
                else:
                    # print("entered final else")

                    data = {}
                    insertLMDB(db_txn, "IdLabelInfo", data)
                print("updateddata after face recognition", data)
                print("\n")

                

    # print("act_batch_res",act_batch_res)
    # print("\n")

    # print("trackIdMemIdDict",trackIdMemIdDict)
    # print(IdLabelInfo)
    
    print("*----------*---------*----------*----------*")

    output_json = conv_jsonnumpy_2_jsoncid(output_json)

    # print(f'old FACE OUTPUT: {output_json}')
    # print("___________________________________________________")
    # print("___________________________________________________")
    # print("___________________________________________________")
    # print("___________________________________________________")

    # print(idsToBeRemoved)
    # print("___________________________________________________")
    # print("___________________________________________________")
    # print("___________________________________________________")
    # print("___________________________________________________")

    output_json['metaData']['object'] = [
        obj for obj in output_json['metaData']['object'] if obj['id'] not in idsToBeRemoved
    ]

    print("\n")
    print(f'updated FACE OUTPUT: {output_json}')
    with open("./static/test.json", "a") as outfile:
        json.dump(output_json, outfile)

    # update cache
    dbpush_members(output_json)


def merge_activity(act_batch_res, output_json):
    if len(output_json['metaData']['object'])>0:
        for obj in output_json['metaData']['object']:
            if int(obj["id"]) in act_batch_res:
                obj["activity"] = act_batch_res[int(obj["id"])]
    return output_json

async def json_publish_activity(primary):
    nc = await nats.connect(servers=nats_urls , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "service.notifications"
    Stream_name = "services"
    ack = await js.publish(Subject, json_encoded)
    print(" ")
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")
    # await asyncio.tim

async def process_results(device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame):
    print("process results")
    subscriptions = device_data[device_id]["subscriptions"]
    # print(output_json)
    act_batch_res={}
    if 'Activity' in subscriptions:
        act_batch_res = activity_main(org_frames_lst,bbox_tensor_lst,obj_id_ref)
        for objId in act_batch_res:
            objId = str(objId)
            with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                data = fetchLMDB(db_txn, "IdLabelInfo")
                # print("data fetched by activity",data)
                # print("\n")

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
                        # data = {}
                        data[objId] = {"memberID":None,"activity":actList}
                        insertLMDB(db_txn, "IdLabelInfo", data)


                else:
                    actList = []
                    actList.append(act_batch_res[int(objId)])
                    data = {}
                    data[objId] = {"memberID":None,"activity":actList}
                    insertLMDB(db_txn, "IdLabelInfo", data)
                print("data after activityupdate ",data)
                print("\n")

                


        # print(act_batch_res)
        output_json = merge_activity(act_batch_res, output_json)

    if output_json['metaData']['object']:
        # print(output_json)
        batchId = str(uuid.uuid4())
        # print(device_data)
        output_json["tenantId"] = device_data[device_id]['tenantId']
        output_json["batchid"] = batchId
        output_json["deviceid"] = device_id_new
        output_json["timestamp"] = str(device_timestamp)
        output_json['geo']['latitude'] = device_data[device_id]['lat']
        output_json['geo']['longitude'] = device_data[device_id]['long']
        # geo = "testing_geo"
        # output_json["geo"] = geo
        if len(output_json["metaData"]["frameAnomalyScore"])>0:
            franaavg = sum(output_json["metaData"]["frameAnomalyScore"])/len(output_json["metaData"]["frameAnomalyScore"])
        else:
            franaavg = 0
        output_json["metaData"]["frameAnomalyScore"] = franaavg
        output_json["metaData"]['detect'] = len(output_json["metaData"]['object'])
        output_json["metaData"]['count']["peopleCount"] = len(output_json["metaData"]['object'])
        output_json["version"] = "v0.0.4"
        if fin_full_frame is not None:
            output_json["metaData"]["cid"] = fin_full_frame

        # for each in output_json['metaData']['object']:
        if len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly])>0 or len([True for each in [each["class"] for each in output_json['metaData']['object']] if each in anamoly_object])>0 or len([True for each in [each["track"] for each in output_json['metaData']['object']] if each in anamolyMemberCategory])>0:
            if "Alarm" in subscriptions:
                print("alarm trigger for activities")
                try:
                    alarm()
                except:
                    print("alarm is not connected / couldn't connect to alarm")
            print("Anomoly check cnt:",len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly]))
            
            output_json["type"] = "anomaly"
            output_json_fr = copy.deepcopy(output_json)
            
            if 'Bagdogra' not in subscriptions:
                # add if sub has data process 
                output_json = conv_jsonnumpy_2_jsoncid(output_json)
                # add if sub has data process 

            for detection in output_json['metaData']['object']:
                del detection['cropsNumpyList']
                
            print("THE OUTPUT JSON STRUCTURE: ",output_json)
            
            if 'Bagdogra' not in subscriptions:
                status = dbpush_activities(output_json)
                if(status == "SUCCESS!!"):
                    # add if sub has data process
                    publish_status = asyncio.run(json_publish_activity(primary=output_json))
                    await publish_status
                    print("DB insertion successful :)")
                    if 'Facial-Recognition' in subscriptions:
                        face_recognition_process(output_json_fr,device_id, act_batch_res)
                        # asyncio.create_task(face_recognition_process(output_json_fr,device_id))
                        # threading.Thread(target = face_recognition_process,args = (output_json_fr,device_id,)).start()
                    #did,track_type = find_person_type(objectt["crop"])

                    # with open("./static/test.json", "a") as outfile:
                    #     json.dump(output_json, outfile)
                elif(status == "FAILURE!!"):
                    print("DB insertion got failed :(")
        else:
            output_json_fr = copy.deepcopy(output_json)
            output_json = conv_jsonnumpy_2_jsoncid(output_json)
            # add if sub has data process 
            for detection in output_json['metaData']['object']:
                del detection['cropsNumpyList']
            # print("THE OUTPUT JSON STRUCTURE: ",output_json)
            
            if 'Bagdogra' not in subscriptions:
                status = dbpush_activities(output_json)
                if(status == "SUCCESS!!"):
                    print("DB insertion successful :)")
                    if 'Facial-Recognition' in subscriptions:
                        face_recognition_process(output_json_fr,device_id, act_batch_res)
                        # asyncio.create_task(face_recognition_process(output_json_fr,device_id))
                        # threading.Thread(target = face_recognition_process,args = (output_json_fr,device_id,)).start()
                    # with open("./static/test.json", "a") as outfile:
                    #     json.dump(output_json, outfile)
                elif(status == "FAILURE!!"):
                    print("DB insertion got failed :(")   

    await asyncio.sleep(0)


async def process_publish(device_id,batch_data,device_data,device_timestamp):
    # print("threading.activeCount",threading.activeCount())

    # with semaphore:
    #     print("threading.activeCount inside semaphore",threading.activeCount())

    # print("entering process_publish")
    global anamoly_object, anamoly
    device_id_new = device_data[device_id]["deviceId"]
    detectss = 0
    fin_full_frame = None
    for each in batch_data:
        if each["total_detects"] > detectss:
            detectss = each["total_detects"]
            # print("detectss: ",detectss)
            # cv2.imwrite("./static/img_max_det.jpg", each["det_frame"][0])
            fin_full_frame = each["det_frame"]
    bbox_tensor_lst = [frame_data["bbox_tensor"] for frame_data in batch_data]
    org_frames_lst = [frame_data["org_frame"] for frame_data in batch_data]
    obj_id_ref = [frame_data["bbox_ref_list"] for frame_data in batch_data]
    # output_func(batch_data,device_id,device_timestamp)
    output_json = output_func(batch_data,device_id,device_timestamp)
    # print(output_json)
    # print(output_json)
    # process_results(device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame)
    # threading.Thread(target = process_results, args = (device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame,)).start()
    # p = Process(target = process_results, args = (device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame,))
    # p.start()
    # p.join()
    # process_results(device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame)
    inference_task = asyncio.create_task(process_results(device_id,batch_data,device_data,device_timestamp, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame))
    await inference_task
# async def intermediate(device_id,batch_data,dev_id_dict, frame_timestamp):
#     print("got imgs")
#     process_publish(device_id,batch_data,dev_id_dict, frame_timestamp)
#     # threading.Thread(target = process_publish,args = (device_id,batch_data,dev_id_dict, frame_timestamp,)).start()

def frame_2_dict():
    while True:
        try:
            # print("got inputs")
            inputt, dev_id_dict = framedata_queue.get()
            # with lock:
            # print(dev_id_dict)
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

            for objectt in inputt["objects"]:
                if "Alarm" in subscriptions:
                    if objectt["detect_type"] in anamoly_object and objectt["age"] > trigger_age:
                        trigger_age = trigger_age + 1
                        
                        print("Alarm triggered for "+objectt["detect_type"]+" age: "+str(objectt["age"]))
                        try:
                            alarm()
                        except:
                            print("alarm is not connected / couldn't connect to alarm")
                    
                obj_dict = {}
                obj_dict[objectt["obj_id"]] = {}
                obj_dict[objectt["obj_id"]]["type"] = objectt["detect_type"]
                obj_dict[objectt["obj_id"]]["activity"] = "No Activity"
                obj_dict[objectt["obj_id"]]["confidence"] = objectt["confidence_score"]
                obj_dict[objectt["obj_id"]]["did"] = None
                obj_dict[objectt["obj_id"]]["track_type"] = None
                # obj_dict[objectt["obj_id"]]["age"] = objectt["age"]
                obj_dict[objectt["obj_id"]]["crops"] = [objectt["crop"]]
                frame_data.append(obj_dict)
                
                if(objectt["detect_type"] == "Male" or objectt["detect_type"] == "Female"):
                    bbox_coord = [objectt["bbox_left"], objectt["bbox_top"], objectt["bbox_right"], objectt["bbox_bottom"]]
                    bbox_ref_list.append(objectt["obj_id"])
                    bbox_list.append(bbox_coord)

            bbox_np = np.array(bbox_list, dtype=np.float32)
            bbox_array = np.reshape(bbox_np, (-1, 4))      
            frame_info_anamoly = anamoly_score_calculator(frame_data)
            frame_data.clear()
            frame_anamoly_wgt = frame_weighted_avg(frame_info_anamoly)

            cidd = [inputt["np_arr"]]
            final_frame = {"frame_id":inputt["frame_number"],"frame_anamoly_wgt":frame_anamoly_wgt,"detection_info":frame_info_anamoly,"cid":cidd, "bbox_tensor":bbox_array, "org_frame":inputt["org_frame"], "bbox_ref_list":bbox_ref_list, "total_detects":inputt["total_detect"], "det_frame":cidd}

            if device_id in isolate_queue:
                isolate_queue[device_id].append(final_frame)
            else:
                isolate_queue[device_id] = []
                isolate_queue[device_id].append(final_frame)

            for each in isolate_queue:
                # print(len(isolate_queue[each]))
                # print("len(isolate_queue[each])",len(isolate_queue[each]))
                # print("len(isolate_queue[each]): ",len(isolate_queue[each]))
                if len(isolate_queue[each])>batch_size:
                    batch_data = isolate_queue[each]
                    isolate_queue[each] = []
                    # intermediate(device_id,batch_data,dev_id_dict, frame_timestamp)
                    asyncio.run(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp))
                    # process_publish(device_id,batch_data,dev_id_dict, frame_timestamp)
                    # asyncio.create_task(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp))
        except Exception as e:
            print(e)


                    









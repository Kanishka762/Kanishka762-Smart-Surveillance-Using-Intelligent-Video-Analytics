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
import threading

from alarm_light_trigger import alarm
from try_anamoly import anamoly_score_calculator, frame_weighted_avg
from project_1_update_ import output_func
from db_insert import dbpush_activities, dbpush_members
from person_type_new import find_person_type
# from warehouse import get_info

import cv2
from mini_mmaction.demo.demo_spatiotemporal_det import activity_main

path = os.getcwd()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")

anamoly_object = ast.literal_eval(os.getenv("anamoly_object"))
anamoly = ast.literal_eval(os.getenv("anamoly"))


# anamoly_object =  ['Fire', 'Smoke','Gun', 'Knife'] #, "Male", "Female"
# anamoly = ["falling down","jumping","lying","running","hitting an object","kicking","shooting/holding a gun","throwing","fighting/beating","kicking"]


batch = []
frame_cnt = 0
isolate_queue = {}
trigger_age = 1

def conv_path2cid(pathh):
    command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(file_path=pathh,ipfs_url=ipfs_url)
    output = sp.getoutput(command)
    return output

def conv_jsonnumpy_2_jsoncid(primary):
    # convert full frame numpy to cid
    # print(primary)
    if not os.path.exists("./ipfs_data"):
        os.makedirs("./ipfs_data")
    if not os.path.exists("./ipfs_data/"+primary['deviceid']):
        os.makedirs("./ipfs_data/"+primary['deviceid'])
    pathh = "./ipfs_data/"+primary['deviceid']+"/cid_ref_full.jpg"
    cv2.imwrite(pathh,primary["metaData"]['cid'][0])
    primary["metaData"]['cid'] = conv_path2cid(pathh)

    for each in primary["metaData"]["object"]:
        
        pathh = "./ipfs_data/"+primary['deviceid']+"/cid_ref.jpg"
        cv2.imwrite(pathh,np.array(each["cids"][0], dtype=np.uint8))
        each["cids"] = conv_path2cid(pathh)
    # pathh = "./ipfs_data/"+primary['deviceid']+"/cid_ref_full.jpg"
    # cv2.imwrite(pathh,np.array(primary["metaData"]['cid'][0], dtype=np.uint8))
    # primary["cid"] = primary["cid"]
    # print(primary)
    return primary


def face_recognition_process(output_json,datainfo,device_id):
    print("started face recognition")
    for detection in output_json['metaData']['object']:
        # image_path = cid_to_image(detection['cids'],device_id)
        crop_image = detection['cids'][0]
        # print("crop_image",crop_image)
        did, track_type = find_person_type(crop_image,datainfo)
        print( did, track_type)
        if did == "":
            did = None
        detection["track"] = track_type
        detection['memDID'] = did
    output_json = conv_jsonnumpy_2_jsoncid(output_json)
    print("\n")
    print(f'FACE OUTPUT: {output_json}')
    with open("test.json", "a") as outfile:
        json.dump(output_json, outfile)


def merge_activity(act_batch_res, output_json):
    if len(output_json['metaData']['object'])>0:
        for obj in output_json['metaData']['object']:
            if int(obj["id"]) in act_batch_res:
                # print(act_batch_res)
                # print(act_batch_res[obj["id"]])
                obj["activity"] = act_batch_res[int(obj["id"])]
            else:
                obj["activity"] = None
    return output_json


def process_results(device_id,batch_data,device_data,device_timestamp, datainfo, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame):
    act_batch_res = activity_main(org_frames_lst,bbox_tensor_lst,obj_id_ref)
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
        if len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly])>0 or len([True for each in [each["class"] for each in output_json['metaData']['object']] if each in anamoly_object])>0:
            alarm()
            print("Anomoly check cnt:",len([True for each in [each["activity"] for each in output_json['metaData']['object']] if each in anamoly]))
            output_json["type"] = "anomaly"
            output_json_fr = copy.deepcopy(output_json)
            output_json = conv_jsonnumpy_2_jsoncid(output_json)
            print("THE OUTPUT JSON STRUCTURE: ",output_json)

            status = dbpush_activities(output_json)
            
            if(status == "SUCCESS!!"):
                asyncio.run(json_publish_activity(primary=output_json))
                print("DB insertion successful :)")
                # face_recognition_process(output_json_fr,datainfo,device_id)
                #did,track_type = find_person_type(objectt["crop"], datainfo)

                with open("test.json", "a") as outfile:
                    json.dump(output_json, outfile)
            elif(status == "FAILURE!!"):
                print("DB insertion got failed :(")
        else:
            output_json_fr = copy.deepcopy(output_json)
            output_json = conv_jsonnumpy_2_jsoncid(output_json)
            print("THE OUTPUT JSON STRUCTURE: ",output_json)
            
            status = dbpush_activities(output_json)
            if(status == "SUCCESS!!"):
                print("DB insertion successful :)")
                # face_recognition_process(output_json_fr,datainfo,device_id)
                with open("test.json", "a") as outfile:
                    json.dump(output_json, outfile)
            elif(status == "FAILURE!!"):
                print("DB insertion got failed :(")   


async def process_publish(device_id,batch_data,device_data,device_timestamp, datainfo):
    global anamoly_object, anamoly
    device_id_new = device_data[device_id]["deviceId"]
    detectss = 0
    fin_full_frame = None
    for each in batch_data:
        if each["total_detects"] > detectss:
            detectss = each["total_detects"]
            print("detectss: ",detectss)
            cv2.imwrite("max_det.jpg", each["det_frame"][0])
            fin_full_frame = each["det_frame"]
    bbox_tensor_lst = [frame_data["bbox_tensor"] for frame_data in batch_data]
    org_frames_lst = [frame_data["org_frame"] for frame_data in batch_data]
    obj_id_ref = [frame_data["bbox_ref_list"] for frame_data in batch_data]
    output_json = output_func(batch_data,device_id,device_timestamp)
    threading.Thread(target = process_results, args = (device_id,batch_data,device_data,device_timestamp, datainfo, org_frames_lst, obj_id_ref, output_json, bbox_tensor_lst, device_id_new,fin_full_frame,)).start()


def frame_2_dict(inputt, dev_id_dict, datainfo):
    global anamoly_object, anamoly
    global frame_cnt
    global trigger_age

    frame_timestamp = inputt["frame_timestamp"]
    frame_cnt += frame_cnt
    frame_data = []
    bbox_list = []
    bbox_ref_list = []

    for objectt in inputt["objects"]:
        if objectt["detect_type"] in anamoly_object and objectt["age"] > trigger_age:
            trigger_age = trigger_age + 1
            print("Alarm triggered for "+objectt["detect_type"]+" age: "+str(objectt["age"]))
            # alarm()
            
        obj_dict = {}
        obj_dict[objectt["obj_id"]] = {}
        obj_dict[objectt["obj_id"]]["type"] = objectt["detect_type"]
        obj_dict[objectt["obj_id"]]["activity"] = "No"
        obj_dict[objectt["obj_id"]]["confidence"] = objectt["confidence_score"]
        obj_dict[objectt["obj_id"]]["did"] = None
        obj_dict[objectt["obj_id"]]["track_type"] = None
        obj_dict[objectt["obj_id"]]["age"] = objectt["age"]
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

    device_id = inputt["camera_id"]
    if device_id in isolate_queue:
        isolate_queue[device_id].append(final_frame)
    else:
        isolate_queue[device_id] = []
        isolate_queue[device_id].append(final_frame)

    for each in isolate_queue:
        if len(isolate_queue[each])>279:
            batch_data = isolate_queue[each]
            isolate_queue[each] = []
            asyncio.run(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp,datainfo))









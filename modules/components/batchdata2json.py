from modules.components.load_paths import *
import cv2
import math
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast
import gc
import os, shutil
import subprocess as sp

# datetime
from pytz import timezone
import time
from datetime import datetime
import numpy as np
collectionCnt = 30

load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")

global id


def select_elements_with_equal_intervals(input_list, collectionCnt):
    if len(input_list)>collectionCnt:
        idx = np.round(np.linspace(0, len(input_list) - 1, collectionCnt)).astype(int)
        return [input_list[i] for i in idx]
    else:
        return input_list

def generateCropList(listOfFrames):
    cropListDict = {}
    finalCropListDict = {}
    for frame in listOfFrames:
        for objectdict in frame['detection_info']:
            for trackId in objectdict:
                if trackId in cropListDict:
                    cropListDict[trackId].append(objectdict[trackId]['crops'])
                else:
                    cropListDict[trackId] = []
                    cropListDict[trackId].append(objectdict[trackId]['crops'])
    # print([{each:len(cropListDict[each])} for each in cropListDict])
    for trackId in cropListDict:
        selectedList = select_elements_with_equal_intervals(cropListDict[trackId], collectionCnt)
        finalCropListDict[trackId] = selectedList
    # print([{each:len(finalCropListDict[each])} for each in finalCropListDict])
    return finalCropListDict

def output_func(my_list,device_id,device_timestamp):
    # print(my_list[0]['detection_info'])
    cropListDict = generateCropList(my_list)
    my_list = [d for d in my_list if d['detection_info'] is not None]
    my_list = [my_list]
    frames=[]               # this variable will hold the frame_id of all the frames in which a atleast one detection was made"
    ids_to_be_monitored=[]  #re-id of all the person type detection with anamoly score>50
    # frame_anamoly_wgt = []
    person_counts = {}
    vehicle_counts = {}
    object_counts = {}
    activity_dict = {}
    det_score_dict = {}
    temp_list=[]
    metaObj = []
    frame_cid = {}
    did = []
    did_dict = {}
   
    most_occurred_activities = {}
    activity_scores= {}
    people_count_list =[]
    vehicle_count_list = []
    object_count_list = []
    crops = {}



    count_non_empty = sum(1 for item in my_list if item != '')
    sub_my_list = my_list[0]
    h=len(my_list[0])
    
    



    last_frame_data = []
    for i in range(1, len(sub_my_list)):
        if len(sub_my_list[h-i]["detection_info"]) > 0:
            last_frame_data = sub_my_list[h-i]["detection_info"]
            break 
        else:
            continue
    lastframe_re_ids = [str(list(each.keys())[0]) for each in last_frame_data]
    cidss_dic =  {}
    
    for each in sub_my_list:
        if len(each['detection_info'])>0:
            for every in each['detection_info']:
                if list(every.keys())[0] not in cidss_dic:
                    cidss_dic[list(every.keys())[0]] = []
                    cidss_dic[list(every.keys())[0]].append(every[list(every.keys())[0]]['crops'])
                else:
                    cidss_dic[list(every.keys())[0]].append(every[list(every.keys())[0]]['crops'])
        if each["cid"]:
            if "frame_cids" not in cidss_dic:
                cidss_dic["frame_cids"] = []
                cidss_dic["frame_cids"].append(each["cid"])
            else:
                cidss_dic["frame_cids"].append(each["cid"])

    final_cid = {}
    for each in cidss_dic:
        if len(cidss_dic[each]) <= 2:
            final_cid[each] = cidss_dic[each][0]
        elif len(cidss_dic[each]) == 0:
            final_cid[each] = None
        elif len(cidss_dic[each]) > 2:
            idxx = round(len(cidss_dic[each])/2)
            final_cid[each] = cidss_dic[each][idxx]

    for x in my_list:
        for item in x:
            # frame_anamoly_wgt.append(item['frame_anamoly_wgt'])
            frame_id = item['frame_id']
            detection_info = item['detection_info']
            if detection_info ==[]:
                continue
            else:
                frames.append(frame_id)
            person = 0
            vehicle = 0
            object = 0
            cid = item['cid']
            for detection in item['detection_info']:
                for key,values in detection.items():
                    re_id = key 
                    did = values.get('did')
#DID dictionary , it holds re_id and list of DIDs corresponding to each re_id
                    if key in did_dict :
                       did_dict[key].append(did)                             
                    else:
                       did_dict[key] = [did]
   

                    if values['type'] == 'Person' and re_id not in people_count_list :
                            people_count_list.append(re_id)
                            person += 1
                            person_counts[frame_id] = person

                    elif values['type'] == 'Vehicle' and re_id not in vehicle_count_list :
                        vehicle_count_list.append(re_id)
                        vehicle +=1
                        vehicle_counts[frame_id] = vehicle
                       
                    elif values['type'] == 'Elephant' and re_id not in object_count_list :
                        object_count_list.append(re_id)
                        object +=1
                        object_counts[frame_id] = object 
              
                    detection_score = float(values.get('anamoly_score') or 0)
                    activity_score =  float(values.get('activity_score') or 0)
                    
                    track = values.get('track_type') or None
                    detect_type = values.get('type')  
                    
                    did  =  did_dict[re_id][0]
                     
                    crop = values.get('crops') 

                    for m in final_cid:
                        if m==re_id:
                            c_id = final_cid[m]
             
                    activity = values.get('activity')

#activity score dictionary holds activity score for each activity
                    activity_scores[activity] = activity_score

#detection score dictionary , it holds re_id and list of detection scores corresponding to each re_id
                    if re_id in det_score_dict :
                        if detection_score not in det_score_dict[re_id]:
                            det_score_dict[re_id].append(detection_score)
                    else:
                        det_score_dict[re_id] = [detection_score]

#activity dictionary , it holds re_id and list of activities corresponding to each re_id
                    if re_id in activity_dict:
                        if activity not in activity_dict[re_id]:
                                activity_dict[re_id].append(activity)
                                
                    else:
                        activity_dict[re_id] = [activity]
                        
#filtering most frequent activity for each re_id from the list of activities
                    for id, activities in activity_dict.items():
                            most_occurred_activity = max(set(activities), key=activities.count)
                            most_occurred_activities[id] = most_occurred_activity
                        
                    
                    act  =   most_occurred_activities[re_id]

                    act_score = activity_scores[act]
             
                    det_score = det_score_dict[re_id]

                 
                    if did == " " or "" or None:
                        memDID = None
                    else:
                        memDID = str(did)

                    temp = {
                                "class": detect_type,
                                "detectionScore": det_score,
                                "activityScore": act_score,
                                "track": track,
                                "id": str(re_id),
                                "memDID" : memDID,
                                "activity": act,
                                "detectTime" : "",
                                "cids": c_id,
                                "cropsNumpyList":cropListDict[re_id]
                            } 
                    temp_list.append(temp)            
    
    for obj in temp_list:
      if not any(x.get("id") == obj.get("id") for x in metaObj):
        metaObj.append(obj)

    for items in metaObj:
        items['detectionScore'] = sum(items['detectionScore'])/len(items['detectionScore'])
        items["detectTime"] = str(device_timestamp)
           
    vehicle_count = sum(vehicle_counts.values())  # this variable will hold the count of total vehicles detected overall
    person_count = sum(person_counts.values())  # this variable will hold the count of total person detected overall    
    animal_count = sum(object_counts.values())  # this variable will hold the count of total elephants detected overall
    total_count = vehicle_count + person_count + animal_count
    
    

    frame_count_vehicle = len(vehicle_counts)  # this variable will hold the count of total frames in which vehicle was detected
    frame_count_person = len(person_counts)    # this variable will hold the count of total frames in which people were detected
    frame_count_animal = len(object_counts)    # this variable will hold the count of total frames in which elephant was detected
    frame_count = len(frames)
    
    
    t_list = [key for elem in my_list for x in elem for detection in x['detection_info'] for key, values in detection.items() if values.get('type') == 'Person' and values.get('anamoly_score') is not None and values['anamoly_score'] > 50]
    for id in t_list:
        if id not in ids_to_be_monitored:
            ids_to_be_monitored.append(id)

    if frame_count_vehicle != 0 and vehicle_count >= frame_count_vehicle:
        avg_Batchcount_vehicle = math.ceil(vehicle_count / frame_count_vehicle)
    else :
        avg_Batchcount_vehicle = 0
    if frame_count_animal != 0 and animal_count >= frame_count_animal:
        avg_Batchcount_animal = math.ceil(animal_count / frame_count_animal)
    else :
        avg_Batchcount_animal = 0
    if frame_count_person !=0 and person_count >= frame_count_person:
       avg_Batchcount_person = math.ceil(person_count / frame_count_person)
    else :
        avg_Batchcount_person = 0
    
    if len(people_count_list) ==1:
        count_p = len(people_count_list)
    else :
        count_p = avg_Batchcount_person

    if len(vehicle_count_list) == 1:
        count_v = len(vehicle_count_list)
    else :
        count_v = avg_Batchcount_vehicle
        
    if len(object_count_list) == 1:
        count_o = len(object_count_list)
    else:
         count_o = avg_Batchcount_animal
    
    count_all = count_o +count_v +count_p

    if 'frame_cids' in final_cid:
        finn = final_cid['frame_cids']
    else:
        finn = None

    metaBatch = {
        "detect": (count_all),
        # "frameAnomalyScore" : frame_anamoly_wgt,
        "count": {"peopleCount": (count_p),
                  "vehicleCount": (count_v),
                  "ObjectCount" : (count_o),
                  },
        "anomalyIds": ids_to_be_monitored,
         'cid' : finn,
        "object": metaObj
       
    }
   
    primary = {
        "type": "activity",
        "deviceid": device_id,
        "batchid": "",
        "timestamp": "",
        "geo": {
            "latitude": "",
            "longitude": ""
        },
        "metaData": metaBatch
    }

    return primary






#import the .env list of subscriptions
#write a if condition for DB
#if that condition satisfies
    #import statements for DB

#write a if condition for activity
#if that condition satisfies
    #import statements for activity model

#write a if condition for obj detection
#if that condition satisfies
    #import statements for obj detection model
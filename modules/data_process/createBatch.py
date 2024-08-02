from modules.components.load_paths import *
from init import loadLogger
import threading
from dotenv import load_dotenv
import numpy as np
import ast
# import multiprocessing
import threading


from modules.alarm.alarm_trigger import sound_alarm
# from modules.data_process.processBatch import process_publish
from modules.deepstream.rtsp2rtsp import framedata_queue

logger = loadLogger()
load_dotenv(dotenv_path)

isolate_queue = {}
batch_size = int(os.getenv("batch_size"))
anamoly_object = ast.literal_eval(os.getenv("anamoly_object"))
anamoly = ast.literal_eval(os.getenv("anamoly"))
trigger_age = 50


# def loadVariableForFrame2Dict(inputt,dev_id_dict):
#     try:
#         global anamoly_object, anamoly
#         global trigger_age
#         device_id = inputt["camera_id"]
#         subscriptions = dev_id_dict[device_id]["subscriptions"]
#         frame_timestamp = inputt["frame_timestamp"]
#         frame_data = []
#         bbox_list = []
#         bbox_ref_list = []
        
#         return anamoly_object, trigger_age, device_id, subscriptions, frame_timestamp, frame_data, bbox_list, bbox_ref_list
#     except Exception as e:
#         logger.error("An error occurred while loading variable for frametodict", exc_info=e)

# def checkForAnomalousObject(objectt, anamoly_object, trigger_age):
#     try:
#         return (
#             objectt["detect_type"] in anamoly_object
#             and objectt["age"] > trigger_age
#         )
#     except Exception as e:
#         logger.error("An error occurred while checking for anomalous object to trigger alarm", exc_info=e)

# def checkForobjectAlarm(objectt, anamoly_object, trigger_age, subscriptions):
#     try:
#         if "Alarm" in subscriptions and checkForAnomalousObject(objectt, anamoly_object, trigger_age):
#             trigger_age = trigger_age + 1
#             print(f"Alarm triggered for {objectt['detect_type']} age: {str(objectt['age'])}")
#             try:
#                 sound_alarm()
#             except Exception:
#                 print("alarm is not connected / couldn't connect to alarm")
#     except Exception as e:
#         logger.error("An error occurred while checking for object alarm", exc_info=e)

# def structureobjdict(object_data):
#     try:
#         return {
#                 object_data["obj_id"]: {
#                     "type": object_data["detect_type"],
#                     "activity": "No Activity",
#                     "confidence": object_data["confidence_score"],
#                     "did": None,
#                     "track_type": None,
#                     "crops": [object_data["crop"]]
#                 }
#             }
#     except Exception as e:
#         logger.error("An error occurred while structing object dict for frametodict", exc_info=e)

# def createBBocCoord(object_data, bbox_ref_list, bbox_list):
#     try:
#         bbox_coord = [object_data["bbox_left"], object_data["bbox_top"], object_data["bbox_right"], object_data["bbox_bottom"]]
#         bbox_ref_list.append(object_data["obj_id"])
#         bbox_list.append(bbox_coord)
#         return bbox_ref_list, bbox_list
#     except Exception as e:
#         logger.error("An error occurred while creating bbox coord for frametodict", exc_info=e)

# def createObjectsDict(inputt, subscriptions, frame_data, bbox_ref_list, bbox_list, trigger_age, anamoly_object):
#     try:
#         if len(inputt["objects"]) > 0:
#             for object_data in inputt["objects"]:
#                 # #object_data['crop']
#                 # reid = func(object_data['crop'])
#                 threading.Thread(target = checkForobjectAlarm,args = (object_data, anamoly_object, trigger_age, subscriptions,)).start()
#                 obj_dict = structureobjdict(object_data)

#                 frame_data.append(obj_dict)

#                 if object_data["detect_type"] in ["Male", "Female"]:
#                     bbox_ref_list, bbox_list = createBBocCoord(object_data, bbox_ref_list, bbox_list)
#         return frame_data, bbox_ref_list, bbox_list
#     except Exception as e:
#         logger.error("An error occurred while creating objects dict for frametodict", exc_info=e)

# def batchingFrames(device_id, finalFrameDict, dev_id_dict, frame_timestamp):
#     try:
#         if device_id not in isolate_queue:
#             isolate_queue[device_id] = []
#         isolate_queue[device_id].append(finalFrameDict)
#         for each in isolate_queue:
#             if len(isolate_queue[each])>batch_size:
#                 batch_data = isolate_queue[each]
#                 isolate_queue[each] = []
#                 #TODO:put in queue
#                 logger.info("sending a batch of frames to process")
#                 # process_publish(device_id,batch_data,dev_id_dict, frame_timestamp)
#                 # asyncio.create_task(process_publish(device_id,batch_data,dev_id_dict, frame_timestamp))
#     except Exception as e:
#         logger.error("An error occurred while batching frames", exc_info=e)

# def structureFinalFrameDict(inputt,frame_data, cidd, bbox_array, bbox_ref_list):
#     try:
#         return{
#             "frame_id":inputt["frame_number"],
#         "detection_info":frame_data,
#         "cid":cidd, 
#         "bbox_tensor":bbox_array, 
#         "org_frame":inputt["org_frame"], 
#         "bbox_ref_list":bbox_ref_list, 
#         "total_detects":inputt["total_detect"], 
#         "det_frame":cidd
#         }
#     except Exception as e:
#         logger.error("An error occurred while structing final frame dict for frametodict", exc_info=e)

def frame_2_dict():
    global trigger_age
    while True:
        try:
            inputt = framedata_queue.get()
            if len(inputt["objects"]) > 0:
                for objectt in inputt["objects"]:
                    if objectt["detect_type"] in anamoly_object and objectt["age"] > trigger_age:
                        # trigger_age = trigger_age + 1
                        logger.info("Alarm triggered for" + objectt["detect_type"]+" age: "+str(objectt["age"]))
                        try:
                            threading.Thread(target=sound_alarm, args=()).start()
                            # multiprocessing.Process(target = sound_alarm, args=()).start()
                            # sound_alarm()
                        except Exception as e:
                            logger.error("alarm is not connected / couldn't connect to alarm", e)
        except Exception as e:
            logger.error("Error in frame_2_dict", e)
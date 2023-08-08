from modules.components.load_paths import *
import os 
from os.path import join, dirname
import ast
from dotenv import load_dotenv
import shutil
import asyncio
import threading
from modules.db.db_push import gst_hls_push
from modules.db.db_fetch_devices import filter_devices
from modules.deepstream.person_model import load_lmdb_list
from modules.db.db_fetch_members import fetch_db_mem
from modules.face_recognition_pack.lmdb_components import load_lmdb_fst
# cwd = os.getcwd()
# data_path = join(cwd, 'data')
# dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

rtsp_links = ast.literal_eval(os.getenv("rtsp_links"))
subscriptions = ast.literal_eval(os.getenv("subscriptions"))

def create_device_dict():
    
    if 'Face_Recognition' in subscriptions:
        load_lmdb_list()
        print("removed lmdb contents")
        mem_data = fetch_db_mem()
        print(mem_data)
        load_lmdb_fst(mem_data)
        load_lmdb_list()
    # print(known_blacklist_id)
    # return True

    device_det = filter_devices()
    dev_details = []

    for i,chunk in enumerate(device_det):
        device_dict = {}
        device_dict["deviceId"] = chunk[0]
        device_dict["tenantId"] = chunk[1]
        device_dict["urn"] = chunk[2]
        device_dict["ddns"] = chunk[3]
        device_dict["ip"] = chunk[4]
        device_dict["port"] = chunk[5]
        device_dict["videoEncodingInformation"] = 'H265'
        device_dict["username"] = chunk[7]
        device_dict["rtsp"] = chunk[8]
        # device_dict["rtsp"] = rtsp_links[i]
        # device_dict["rtsp"] = "rtsp://happymonk:admin123@streams.ckdr.co.in:4554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
        device_dict["password"] = chunk[9]
        device_dict["subscriptions"] = chunk[10]
        device_dict["lat"] = chunk[11]
        device_dict["long"] = chunk[12]
        dev_details.append(device_dict)
            
        # if i == 0:
        #     break    
       
    threading.Thread(target=gst_hls_push,args=(dev_details,)).start()
    return dev_details
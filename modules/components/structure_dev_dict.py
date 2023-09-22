from modules.components.load_paths import *
from init import loadLogger


import os 
from os.path import join, dirname
import ast
from dotenv import load_dotenv
import shutil
import asyncio
import threading
import queue


from modules.db.db_push import gst_hls_push
from modules.db.db_fetch_devices import filter_devices
from modules.db.db_fetch_members import fetch_db_mem
from modules.components.rtsp_check import check_rtsp_stream
from modules.face_recognition_pack.membersSubscriber import startMemberService
from modules.face_recognition_pack.recog_objcrop_face import load_lmdb_fst

mem_data_queue = queue.Queue()
logger = loadLogger()

load_dotenv(dotenv_path)

rtsp_links = ast.literal_eval(os.getenv("rtsp_links"))

def create_device_dict():
    try:
        device_det = filter_devices()
        dev_details = []

        for i,chunk in enumerate(device_det):
            device_dict = {
                "deviceId": chunk[0],
                "tenantId": chunk[1],
                "urn": chunk[2],
                "ddns": chunk[3],
                "ip": chunk[4],
                "port": chunk[5],
                "videoEncodingInformation": 'H265',
                "username": chunk[7],
                "rtsp": "file:///home/srihari/facerecog.mp4",
                "password": chunk[9],
                "subscriptions": [
                    'Activity',
                    'Fire/Smoke',
                    'Dangerous-Object'
                ],
                "lat": chunk[11],
                "long": chunk[12],
            }
            dev_details.append(device_dict)
            if (i==0):
                break
    except Exception as e:
        logger.error("An error occurred while structuring device details", exc_info=e)

    try:
        for devs in dev_details:
            if 'Facial-Recognition' in devs["subscriptions"]:
                threading.Thread(target = startMemberService, args=(mem_data_queue,)).start()
                threading.Thread(target=load_lmdb_fst, args=(mem_data_queue,)).start()
                mem_data = fetch_db_mem()
                mem_data_queue.put(mem_data)

        threading.Thread(target=gst_hls_push,args=(dev_details,)).start()
        return dev_details
    except Exception as e:
        logger.error("An error occurred while loading members data", exc_info=e)



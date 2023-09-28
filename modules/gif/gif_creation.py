from modules.components.load_paths import *
from init import loadLogger
import os
from os.path import join, dirname
from dotenv import load_dotenv
# from modules.db.db_push import gif_push
import asyncio
import threading
import imageio
import cv2
import subprocess as sp
import queue
# import multiprocessing 

result_queue = queue.Queue()
load_dotenv(dotenv_path)
ipfs_url = os.getenv("ipfs")
logger = loadLogger()

gif_dict = {}
gif_created = {}

def create_gif_dictionary(dev_id_dict):
    try:
        for key, value in dev_id_dict.items():
            device_id = value['deviceId']
            if device_id not in gif_dict:
                gif_dict[device_id] = []
            if device_id not in gif_created:
                gif_created[device_id] = False
        return gif_dict, gif_created
    except Exception as e:
        logger.error("An error occurred while creating gif dictionary", exc_info=e)

def call_gif_push(path, device_meta, gif_batch):
    print("LENGTH OF THE BATCH: ", len(gif_batch))
    with imageio.get_writer(path, mode="I") as writer:
        for idx, frame in enumerate(gif_batch):
            print("FRAME: ", idx)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

    command = f'ipfs --api={ipfs_url} add {path} -Q'
    gif_cid = sp.getoutput(command)
    print("GIF CID: ", gif_cid)
    asyncio.run(intermediate(path, device_meta, gif_batch, gif_cid))

async def intermediate(path, device_meta, gif_batch,gif_cid ):
    result_queue.put([path, device_meta, gif_batch, gif_cid])
    
def gif_build(img_arr, device_meta, gif_dict, gif_created):
    device_id = device_meta['deviceId']
    # filename for gif
    video_name_gif = f'{gif_path}/{str(device_id)}'
    if not os.path.exists(video_name_gif):
        os.makedirs(video_name_gif, exist_ok=True)
    path = f'{video_name_gif}/camera.gif'

    if gif_created[device_id] == False:
        gif_dict[device_id].append(img_arr)
        len_dict = len(gif_dict[device_id])
        print(len_dict)
        if(len_dict == 40):
            print("LENGTH OF DICTIONARY: ",len_dict)
            gif_created[device_id] = True
            call_gif_push(path, device_meta, gif_dict[device_id][-100:])
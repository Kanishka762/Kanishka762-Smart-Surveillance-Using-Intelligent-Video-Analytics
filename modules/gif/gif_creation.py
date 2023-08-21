from modules.components.load_paths import *
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
# cwd = os.getcwd()

# static_path = join(cwd, 'static')
# gif_path = join(static_path, 'Gif_output')

# data_path = join(cwd, 'data')
# dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")


# # gif_path = os.getenv("path_gif")
# print(gif_path)
# if os.path.exists(gif_path) is False:
#     os.mkdir(gif_path)
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
    # gifstatus = asyncio.create_task(gif_push(result_queue))
    # await gifstatus
def gif_build(img_arr, device_meta, gif_dict, gif_created):
    
    
    device_id = device_meta['deviceId']
      
    # filename for gif
    video_name_gif = gif_path + '/' + str(device_id)
    if not os.path.exists(video_name_gif):
        os.makedirs(video_name_gif, exist_ok=True)
        
    path = video_name_gif + '/' + 'camera.gif'
    
    if gif_created[device_id] == False:
        gif_dict[device_id].append(img_arr)
        len_dict = len(gif_dict[device_id])

        print(len_dict)
    
        if(len_dict == 40):
            print("LENGTH OF DICTIONARY: ",len_dict)
            gif_created[device_id] = True
            # asyncio.run(call_gif_push(path, device_meta, gif_dict[device_id][-100:]))
            # asyncio.create_task(gif_push(path, device_meta, gif_dict[device_id][-100:]))
            call_gif_push(path, device_meta, gif_dict[device_id][-100:])
            # threading.Thread(target=call_gif_push,args=(path, device_meta, gif_dict[device_id][-100:],)).start()
            
        # gif_push(path, device_meta, gif_dict[device_id][-100:])
        
    
    # print(f"DEVICE ID:{device_id} ---------->  LENGTH OF DEVICE ID:{len_dict}")
    
    # if(skip_dict[device_id] < 300 and skip_dict[device_id]>200):
    #     img_arr_cp = img_arr.copy()
    #     gif_dict[device_id].append(img_arr)
    # elif(skip_dict[device_id] == 300):
    #     threading.Thread(target=gif_push,args=(path, device_info, gif_dict[device_id]),).start()
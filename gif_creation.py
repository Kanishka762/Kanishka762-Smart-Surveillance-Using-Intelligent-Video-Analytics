import os
from dotenv import load_dotenv
from db_push import gif_push
from os.path import join, dirname
import asyncio

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# gif_path = "./Gif_output"
gif_path = os.getenv("path_gif")
if os.path.exists(gif_path) is False:
    os.mkdir(gif_path)

async def gif_build(img_arr, device_meta, gif_dict, gif_created):
    
    device_id = device_meta['deviceId']
      
    # filename for gif
    video_name_gif = gif_path + '/' + str(device_id)
    if not os.path.exists(video_name_gif):
        os.makedirs(video_name_gif, exist_ok=True)
        
    path = video_name_gif + '/' + 'camera.gif'
    
    if gif_created[device_id] == False:
        gif_dict[device_id].append(img_arr)
        len_dict = len(gif_dict[device_id])
    
        if(len_dict == 200):
            print("LENGTH OF DICTIONARY: ",len_dict)
            gif_created[device_id] = True
            asyncio.create_task(gif_push(path, device_meta, gif_dict[device_id][-100:]))
            
        # gif_push(path, device_meta, gif_dict[device_id][-100:])
        
    
    # print(f"DEVICE ID:{device_id} ---------->  LENGTH OF DEVICE ID:{len_dict}")
    
    # if(skip_dict[device_id] < 300 and skip_dict[device_id]>200):
    #     img_arr_cp = img_arr.copy()
    #     gif_dict[device_id].append(img_arr)
    # elif(skip_dict[device_id] == 300):
    #     threading.Thread(target=gif_push,args=(path, device_info, gif_dict[device_id]),).start()
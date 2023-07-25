import os
from os.path import join, dirname
from dotenv import load_dotenv
from modules.db.db_push import gif_push
import asyncio
import threading

cwd = os.getcwd()

static_path = join(cwd, 'static')
gif_path = join(static_path, 'Gif_output')

data_path = join(cwd, 'data')
dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)


# gif_path = os.getenv("path_gif")
print(gif_path)
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
            # asyncio.create_task(gif_push(path, device_meta, gif_dict[device_id][-100:]))
            threading.Thread(target=gif_push,args=(path, device_meta, gif_dict[device_id][-100:],)).start()
            
        # gif_push(path, device_meta, gif_dict[device_id][-100:])
        
    
    # print(f"DEVICE ID:{device_id} ---------->  LENGTH OF DEVICE ID:{len_dict}")
    
    # if(skip_dict[device_id] < 300 and skip_dict[device_id]>200):
    #     img_arr_cp = img_arr.copy()
    #     gif_dict[device_id].append(img_arr)
    # elif(skip_dict[device_id] == 300):
    #     threading.Thread(target=gif_push,args=(path, device_info, gif_dict[device_id]),).start()
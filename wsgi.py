import os 
from os.path import join, dirname
import ast
from dotenv import load_dotenv
import shutil
import asyncio
import threading 

from warehouse import main
from testing_tenant import filter_devices
from db_push import gst_hls_push

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

rtsp_links = ast.literal_eval(os.getenv("rtsp_links"))

if __name__ == '__main__':
    # sys.exit(main(sys.argv))

    if os.path.exists("./Hls_output"):
        # Delete Folder code
        shutil.rmtree("./Hls_output")
    
    if os.path.exists("./Gif_output"):
        # Delete Folder code
        shutil.rmtree("./Gif_output")
    
    # load_lmdb_list()
    # print("removed lmdb contents")
    # mem_data = fetch_db_mem()
    # print(mem_data)
    # load_lmdb_fst(mem_data)
    # load_lmdb_list()

    device_det = filter_devices()
    # print(device_det)
    dev_details = [""]
    for i,chunk in enumerate(device_det):
        device_dict = {}
        device_dict["deviceId"] = chunk[0]
        device_dict["tenantId"] = chunk[1]
        device_dict["urn"] = chunk[2]
        device_dict["ddns"] = chunk[3]
        device_dict["ip"] = chunk[4]
        device_dict["port"] = chunk[5]
        # device_dict["videoEncodingInformation"] = chunk[6]
        device_dict["videoEncodingInformation"] = 'H265'
        device_dict["username"] = chunk[7]
        # device_dict["rtsp"] = rtsp_links[i]
        device_dict["rtsp"] = "rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
        # device_dict["rtsp"] = dev_list[index]
        device_dict["password"] = chunk[9]
        device_dict["subscriptions"] = chunk[10]
        device_dict["lat"] = chunk[11]
        device_dict["long"] = chunk[12]
        # if device_dict["rtsp"].find("rtsp://") != 0:
        #     device_dict["rtsp"] = device_dict["rtsp"]
        dev_details.append(device_dict)
        if i == 1:
            break
    threading.Thread(target=gst_hls_push,args=(dev_details,)).start()
    # asyncio.create_task(gst_hls_push(dev_details))
    main(dev_details)


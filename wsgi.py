import threading
import multiprocessing
import queue
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

from init import *
from modules.components.load_paths import *
from modules.deepstream.rtsp2frames import main
from modules.components.structure_dev_dict import create_device_dict
from modules.db.db_push import gif_push
from modules.data_process.frame_data_process import frame_2_dict

load_dotenv(dotenv_path)
# global_lock = threading.Lock()
#/home/srihari/deepstreambackend/modules/face_recognition_pack/membersSubscriber.py

mem_data_queue = queue.Queue()
rtsp_port = os.getenv("RTSP_PORT")
if rtsp_port is not None and rtsp_port.isdigit():
    rtsp_port_int = int(rtsp_port)

if __name__ == '__main__':
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_int
    server.attach(None)
    
    threading.Thread(target = gif_push).start()
    threading.Thread(target = frame_2_dict).start()
    # threading.Thread(target = merge_global).start()


    # p.daemon=True
    # p.start()
    dev_details = create_device_dict()
    print(dev_details)
    main(dev_details)


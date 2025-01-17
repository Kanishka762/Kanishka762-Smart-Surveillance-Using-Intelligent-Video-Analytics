
import gi
gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

import threading
import queue
from dotenv import load_dotenv

from init import *
from modules.components.load_paths import *
from modules.deepstream.rtsp2rtsp import main
from modules.components.structure_dev_dict import create_device_dict
from modules.data_process.createBatch import frame_2_dict

load_dotenv(dotenv_path)

mem_data_queue = queue.Queue()

logger = loadLogger()

def createRTSPPort():  # sourcery skip: extract-method
    try:
        logger.info("creating server for RTSP")
        rtsp_port = os.getenv("rtsp_port")
        logger.info(rtsp_port)
        if rtsp_port is not None and rtsp_port.isdigit():
            rtsp_port_int = int(rtsp_port)
            server = GstRtspServer.RTSPServer.new()
            server.props.service = "%d" % rtsp_port_int
            server.attach(None)
            logger.info(f"created server for RTSP in port {rtsp_port}")
            return server
        else:
            logger.info(f"couldnot created server for RTSP in port {rtsp_port}")
    except Exception as e:
        logger.error(f"couldnot created server for RTSP in port {rtsp_port}", exc_info=e)



if __name__ == '__main__':
    try:
        server = createRTSPPort()
    except Exception as e:
        logger.error("An error occurred outside function createRTSPPort", exc_info=e)

    try:
        threading.Thread(target = frame_2_dict).start()
    except Exception as e:
        logger.error("An error occurred while creating thread for frame_2_dict", exc_info=e)

    try:
        dev_details = create_device_dict()
        print(dev_details,"This is dev_details")
    except Exception as e:
        logger.error("An error occurred outside function create_device_dict", exc_info=e)

    try:
        main(server, dev_details)
    except Exception as e:
        logger.error("An error occurred outside function main", exc_info=e)


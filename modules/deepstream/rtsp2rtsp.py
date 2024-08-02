from modules.components.load_paths import *
from init import loadLogger

import sys
sys.path.append('../')
import ast
import os
import gi
import configparser
gi.require_version('Gst', '1.0')
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
# import sys
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import datetime
import pyds
import pytz
import numpy as np
import cv2
from dotenv import load_dotenv
import json
import queue
import threading
import math

from modules.components.subelements import findClassList
from modules.deepstream.callBackSubmodules import *
from modules.deepstream.elementsSubmodules import *

logger = loadLogger()
load_dotenv(dotenv_path)

device = os.getenv("device")
tenant_name = os.getenv("tenant_name")
ddns_name = os.getenv("DDNS_NAME")
place = os.getenv("place")
anomaly_objs = ast.literal_eval(os.getenv("anamoly_object"))

MAX_NUM_SOURCES = int(os.getenv("MAX_NUM_SOURCES"))
OSD_PROCESS_MODE = int(os.getenv("OSD_PROCESS_MODE"))
OSD_DISPLAY_TEXT = int(os.getenv("OSD_DISPLAY_TEXT"))

leaky = int(os.getenv("leaky"))
max_size_buffers = int(os.getenv("max_size_buffers"))
max_size_bytes = int(os.getenv("max_size_bytes"))
max_size_time = int(os.getenv("max_size_time"))
max_latency = int(os.getenv("max_latency"))
sync_inputs = int(os.getenv("sync_inputs"))
width = int(os.getenv("width"))
height = int(os.getenv("height"))
batched_push_timeout = int(os.getenv("batched_push_timeout"))
live_source = int(os.getenv("live_source"))
rtsp_bitrate = int(os.getenv("rtsp_bitrate"))
control_rate = int(os.getenv("control_rate"))
vbv_size = int(os.getenv("vbv_size"))
insert_sps_pps = int(os.getenv("insert_sps_pps"))
iframeinterval = int(os.getenv("iframeinterval"))
maxperf_enable = bool(os.getenv("maxperf_enable"))
idrinterval = int(os.getenv("idrinterval"))
preset_level = int(os.getenv("preset_level"))
host = os.getenv("host")
rtsp_port = int(os.getenv("rtsp_port"))
UDP_PORT = int(os.getenv("udp_port"))
host = os.getenv("host")
codec = os.getenv("codec")
async_mode = bool(os.getenv("async"))
qos = int(os.getenv("qos"))
sync = int(os.getenv("sync"))
buffer_size = int(os.getenv("buffer_size"))
clock_rate = int(os.getenv("clock_rate"))
payload = int(os.getenv("payload"))
mtu = int(os.getenv("mtu"))
rgba_format = os.getenv("rgba_format")
model_width = os.getenv("model_width")
model_height = os.getenv("model_height")
osd_width = os.getenv("osd_width")
osd_height = os.getenv("osd_height")
enc_width = os.getenv("enc_width")
enc_height = os.getenv("enc_height")
enc_format = os.getenv("enc_format")
target_duration = int(os.getenv("target_duration"))
playlist_length = int(os.getenv("playlist_length"))
max_files = int(os.getenv("max_files"))
hls_bitrate = int(os.getenv("hls_bitrate"))
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720

timezone = pytz.timezone(f'{place}')  #assign timezone
pgie1_path = "/home/development/deepstreambackend_bagdogra/models/primary/config.txt"

frame_path = f"{cwd}/static/frames/"
infer_path = f"{cwd}/static/image/"

framedata_queue = queue.Queue()

dev_id_dict = {}
dev_status = {}

# past_tracking_meta=[0]

streammux = None
pipeline = None

def tracker_src_pad_buffer_probe(pad,info,dev_list):
    try:
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.info("Unable to get GstBuffer ")
            return Gst.PadProbeReturn.OK
    except Exception as e:
        logger.error("An error occurred while getting GstBuffer", exc_info=e)

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    except Exception as e:
        logger.error("An error occurred while getting Batchmeta", exc_info=e)

    try:
        l_frame = batch_meta.frame_meta_list
    except Exception as e:
        logger.error("An error occurred while getting FrameMeta list", exc_info=e)

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data) #new frame
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except StopIteration as e:
            logger.error("An error occurred while getting FrameMeta and n_frame", exc_info=e)
            break

        try:
            frame_number = frame_meta.frame_num
            camera_id = frame_meta.pad_index
            num_detect = frame_meta.num_obj_meta
            l_obj=frame_meta.obj_meta_list
            device_timestamp = datetime.datetime.now(timezone)
        except Exception as e:
            logger.error("An error occurred while getting FrameMeta information", exc_info=e)  

        try:
            n_frame = frame_color_convert(n_frame, "BGR2RGB")
            frame_copy = np.array(n_frame, copy = True, order = 'C')
            frame_copy = frame_color_convert(frame_copy, "RGBA2RGB")
            # cv2.imwrite(f'{frame_path}/{frame_number}.jpg',frame_copy)
        except Exception as e:
            logger.error("An error occurred while trying to color convert n_frame/frame_copy", exc_info=e)  

        # try:
        #     subscriptions = fetch_subsriptions(dev_id_dict, camera_id)
        #     subscriptions_class_list = findClassList(subscriptions)
        # except Exception as e:
        #     logger.error("An error occurred while trying to fetch Subscription list", exc_info=e)

        try:
            frame_dict = initial_frame_dict(frame_number, num_detect, camera_id, device_timestamp)
        except Exception as e:
            logger.error("An error occurred in creating initial frame_dict", exc_info=e)
            
        n_frame_bbox = None
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data) #new obj
            except StopIteration as e:
                logger.error("An error occurred while getting Object Meta", exc_info=e)
                break

            try:
                confidence_score, detect_type, rect_params, text_params, parent, left, top, width, height, bbox_color = get_object_info(obj_meta)
                # print(detect_type)
            except Exception as e:
                logger.error("An error occurred while getting Object Meta information", exc_info=e)
                
            try:
                bbox_color = fetch_bbox_color(detect_type) 
                update_bbox_color(bbox_color, rect_params)
            except Exception as e:
                logger.error("An error occurred while updating bbox_color", exc_info=e)

            try:
                obj_id = parent.object_id if parent is not None else int(obj_meta.object_id)
                n_frame_bbox = update_frame_label(obj_meta, [], detect_type, text_params, obj_id, frame_copy, bbox_color, frame_number)
                # if n_frame_bbox is not None:
                #     cv2.imwrite(f'/home/development/deepstreambackend_bagdogra/static/image/{frame_number}.jpg',n_frame_bbox)
            except Exception as e:
                logger.error("An error occurred while updating frame labels", exc_info=e) 

            try:
                frame_dict = create_obj_dict(n_frame, obj_meta, obj_id, detect_type, confidence_score, left, top, width, height, frame_dict, frame_number)
            except Exception as e:
                logger.error("An error occurred while creating object dictionary", exc_info=e)

            try: 
                l_obj=l_obj.next
            except StopIteration as e:
                logger.error("An error occurred while iterating next Object", exc_info=e)
                break
        try:
            l_frame=l_frame.next
        except StopIteration as e:
            logger.error("An error occurred while iterating next Frame", exc_info=e)
            break
        try:
            frame_dict['np_arr'] = n_frame_bbox if n_frame_bbox is not None else frame_copy
            frame_dict['org_frame'] = n_frame
            print("Starting to put elements")
            # print("#############", frame_dict["objects"])

            framedata_queue.put(frame_dict)
            # print(f"frame_dict: {frame_dict}, dev_id_dict: {dev_id_dict}")
            if is_aarch64():
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except Exception as e:
            logger.error("An error occurred while putting elements in queue", exc_info=e)
    return Gst.PadProbeReturn.OK

def stop_release_pad(source_id, streammux, pipeline, dev_status):
    try:
        pad_name = "sink_%u" % source_id
        logger.info(pad_name)
        #Retrieve sink pad to be released
        sinkpad = streammux.get_static_pad(pad_name)
        #Send flush stop event to the sink pad, then release from the streammux
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        #Remove the source bin from the pipeline
        pipeline.remove(dev_status[source_id][1])
    except Exception as e:
        logger.error("An error occurred while releasing and removing the source bin from pipeline", exc_info=e)
 
def stop_release_source(source_id):
    global streammux
    global pipeline
    global dev_status
    
    try:
        #Attempt to change status of source to be released 
        state_return = dev_status[source_id][1].set_state(Gst.State.NULL)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            logger.info("STATE CHANGE SUCCESS\n")
            stop_release_pad(source_id, streammux, pipeline, dev_status)
        elif state_return == Gst.StateChangeReturn.FAILURE:
            logger.info("STATE CHANGE FAILURE\n")
        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = dev_status[source_id][1].get_state(Gst.CLOCK_TIME_NONE)
            logger.info("STATE CHANGE ASYNC\n", state_return)
            stop_release_pad(source_id, streammux, pipeline, dev_status)
    except Exception as e:
        logger.error("An error occurred while stopping and releasing source", exc_info=e)
 
def delete_sources(delete_dev):
    global dev_id_dict
    try:
        deviceId = delete_dev['deviceId']
        source_id = next(
            (
                key
                for key, value in dev_id_dict.items()
                if value['deviceId'] == deviceId
            ),
            None,
        )
    except Exception as e:
        logger.error("An error occurred while fetching source_id to delete", exc_info=e)
    
    try:
        logger.info("SOURCE ENABLED: ", dev_status[source_id][0])
        dev_status[source_id][0] = False
        #Release the source
        logger.info("Calling Stop %d " % source_id)
        stop_release_source(source_id)
        if(dev_status[source_id][0]) == False:
            dev_status[source_id][1] = None
    except Exception as e:
        logger.error("An error occurred while calling the source release", exc_info=e)
        
    # #Quit if no sources remaining
    # if (g_num_sources == 0):
    #     loop.quit()
    #     logger.info("All sources stopped quitting")
    #     return False
    return True

def create_add_incoming_source_bin(source_id, add_dev, dev_status):
    try:
        logger.info("Adding a new device %d " % source_id)
        uri_name = add_dev['rtsp']
        dev_id_dict[source_id] = add_dev

        source_bin=create_source_bin(source_id, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %source_id
        sinkpad= streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
        logger.info(dev_status)
        dev_status[source_id][0] = True
        dev_status[source_id][1] = source_bin
        logger.info(dev_status)
    except Exception as e:
        logger.error("An error occurred while creating a new source bin", exc_info=e)

def add_incoming_source(dev_status, source_id, add_dev):
    try:
        for key, value in dev_status.items():
            if not value[0]:
                source_id = key
                logger.info(dev_status)
                break
        if source_id is None:
            source_id = len(dev_status)
            dev_status[source_id] = [None, None]
            logger.info(dev_status)
    except Exception as e:
        logger.error("An error occurred while fetching source_id to add", exc_info=e)

    try:
        create_add_incoming_source_bin(source_id, add_dev, dev_status)
    except Exception as e:
        logger.error("An error occurred while adding a new source bin", exc_info=e)
    
def add_sources(add_dev):
    global dev_status
    try:
        if ((len(dev_status) != 0) and (len(dev_status) < MAX_NUM_SOURCES)):
            source_id = None
            add_incoming_source(dev_status, source_id, add_dev)
        else:
            logger.info("DEVICE LIST IS EITHER EMPTY OR EXCEEDED THE LIMIT!!")
            logger.info("DEVICE ADDITION FAILED")
    except Exception as e:
        logger.error("An error occurred while adding a incoming source", exc_info=e)

TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720

def main(server, args):
    print(args)
    global dev_id_dict
    
    number_sources=len(args)
    print(number_sources)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name = args[i]
        

        if uri_name.find("rtsp://") == 0 :
            is_live = True
        else:
            uri_name = "file://"+uri_name
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

 
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvideoconvert \n")
        
    print("Creating Pgie \n ")
    pgie1 = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie1:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvdsosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    print("Creating tiler \n ")
    nvtiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not nvtiler:
        sys.stderr.write(" Unable to create tiler \n")
    
    print("Creating encoder \n ")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder") 
    if not encoder:
        sys.stderr.write(" Unable to create encoder \n")
    
    print("Creating mpeg-ts muxer \n ")
    container = Gst.ElementFactory.make("mpegtsmux", "mux")
    if not container:
        sys.stderr.write(" Unable to create container \n")

    print("Creating parser \n ")
    parser = Gst.ElementFactory.make("h264parse", "parser") 
    if not parser:
        sys.stderr.write(" Unable to create parser \n")

    tracker = Gst.ElementFactory.make("nvtracker","tracker")
    pipeline.add(tracker)

    demux = Gst.ElementFactory.make("nvstreamdemux","demux")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert","nvvidconv")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvideoconvert \n")

    capsfilter0 = Gst.ElementFactory.make("capsfilter", "capsfilter0")
    if not capsfilter0:
        sys.stderr.write(" Unable to create capsfilter0 \n")

    caps0 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter0.set_property("caps", caps0)


    
    print("Creating capsfilter \n")

    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        sys.stderr.write(" Unable to create capsfilter0 \n")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720")
    capsfilter.set_property("caps", caps)

    capsfilter_osd = Gst.ElementFactory.make("capsfilter", "caps_osd")
    caps_osd = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720")
    capsfilter_osd.set_property("caps", caps_osd)

    queue = Gst.ElementFactory.make("queue", "queue_0")
    queue1 = Gst.ElementFactory.make("queue", "queue_1")
    queue2 = Gst.ElementFactory.make("queue", "queue_2")
    queue3 = Gst.ElementFactory.make("queue", "queue_3")
    queue4 = Gst.ElementFactory.make("queue", "queue_4")
    queue5 = Gst.ElementFactory.make("queue", "queue_5")

    # streammux.set_property('config-file-path', 'mux_config.txt')
    # streammux.set_property('batch-size', number_sources)
    # streammux.set_property('sync-inputs', 1)
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 40000)
    # streammux.set_property('live-source', 1)

    config = configparser.ConfigParser()
    config.read(os.path.join(data_path, 'dstest2_tracker_config.txt'))
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)

    pgie1.set_property('config-file-path', pgie1_path)
    pgie1_batch_size=pgie1.get_property("batch-size")
    if(pgie1_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie1_batch_size," with number of sources ", number_sources," \n")
    pgie1.set_property("batch-size", number_sources)

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    # tiler.set_property("show-source", 2)
    

    print("Creating nvvidconv \n ")
    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        # streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        
    if is_aarch64():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    if not sink:
        sys.stderr.write(" Unable to create sink element \n")

    sink.set_property("qos",0)

    #adding elements to the pipeline
    print("Adding elements to Pipeline \n")
    pipeline.add(pgie1)
    pipeline.add(queue)
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter0)
    pipeline.add(demux)
    pipeline.add(nvosd)
    pipeline.add(tiler)
    pipeline.add(sink)
    
    #linking all the elements
    print("Linking elements in the Pipeline \n")
    streammux.link(queue)
    queue.link(nvvidconv)
    nvvidconv.link(capsfilter0)
    capsfilter0.link(queue1)
    # pgie1.link(queue1)
    queue1.link(pgie1)
    pgie1.link(tracker)
    tracker.link(queue2)
    queue2.link(tiler)
    tiler.link(queue4)
    queue4.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    
    tracker_src_pad=tracker.get_static_pad("src")
    if not tracker_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        # multiprocessing.Process(target = QT.main, args=(devices,)).start()
        print("################################")
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)


    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
        
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
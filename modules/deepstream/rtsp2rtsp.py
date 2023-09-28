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
import datetime
import pyds
import pytz
import numpy as np
import cv2
from dotenv import load_dotenv
import json
import queue
import threading

from modules.gif.gif_creation import gif_build, create_gif_dictionary
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

timezone = pytz.timezone(f'{place}')  #assign timezone
pgie1_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/gender/config.txt"
pgie2_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/fire/config.txt"
sgie1_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/face_detect/config.txt"
sgie2_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/dangerous_object/config.txt"
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

        try:
            subscriptions = fetch_subsriptions(dev_id_dict, camera_id)
            subscriptions_class_list = findClassList(subscriptions)
        except Exception as e:
            logger.error("An error occurred while trying to fetch Subscription list", exc_info=e)

        try:
            gif_dict, gif_created = create_gif_dictionary(dev_id_dict)
            if 'Bagdogra' not in subscriptions:
                threading.Thread(target = gif_build,args = (frame_copy, dev_id_dict[camera_id], gif_dict, gif_created,)).start()
        except Exception as e:
            logger.error("An error occurred in GIF dictionary creation", exc_info=e)

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
            except Exception as e:
                logger.error("An error occurred while getting Object Meta information", exc_info=e)
                
            try:
                bbox_color = fetch_bbox_color(detect_type) 
                update_bbox_color(bbox_color, rect_params)
            except Exception as e:
                logger.error("An error occurred while updating bbox_color", exc_info=e)

            try:
                obj_id = parent.object_id if parent is not None else int(obj_meta.object_id)
                n_frame_bbox = update_frame_label(obj_meta, subscriptions, detect_type, text_params, obj_id, frame_copy, bbox_color, frame_number)
                if n_frame_bbox is not None:
                    cv2.imwrite(f'{infer_path}/{frame_number}.jpg',n_frame_bbox)
            except Exception as e:
                logger.error("An error occurred while updating frame labels", exc_info=e) 

            try:
                frame_dict = create_obj_dict(n_frame, obj_meta, obj_id, detect_type, confidence_score, left, top, width, height, frame_dict, subscriptions_class_list, frame_number)
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
            # print("Starting to put elements")
            framedata_queue.put([frame_dict,dev_id_dict])
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

def main(server, args):
    global dev_id_dict
    global dev_status
    global pipeline
    global streammux
    
    try:
        # Check input arguments
        # past_tracking_meta[0]=1
        logger.info(args)
        number_sources=len(args)
        logger.info("NUMBER OF SOURCES: %d", number_sources)
    except Exception as e:
        logger.error("An error occurred while checking input arguments", exc_info=e)
        
    try:
        if number_sources > 0:
            # Standard GStreamer initialization
            GObject.threads_init()
            Gst.init(None)
            
            try:
                # Create gstreamer elements */
                # Create Pipeline element that will form a connection of other elements
                logger.info("Creating Pipeline \n ")
                pipeline = Gst.Pipeline()
                if not pipeline:
                    sys.stderr.write(" Unable to create Pipeline \n")
            except Exception as e:
                logger.error("An error occurred while creating Pipeline", exc_info=e)
            
            try:
                logger.info("Creating streamux \n ")
                # Create nvstreammux instance to form batches from one or more sources.
                streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
                if not streammux:
                    sys.stderr.write(" Unable to create NvStreamMux \n")
                pipeline.add(streammux)
            except Exception as e:
                logger.error("An error occurred while creating NvStreamMux", exc_info=e)
                
            try:
                for i in range(number_sources):
                    if(len(dev_status)) < MAX_NUM_SOURCES:
                        logger.info("Creating source_bin %d \n", i)
                        uri_name = args[i]['rtsp']
                        dev_id_dict[i] = args[i]
                        dev_status[i] = [None, None]
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
                        dev_status[i] = [True, source_bin]
                    else:
                        logger.info("DEVICE LIMIT IS EXCEEDED!! CANNOT ADD ANYMORE DEVICES")
            except Exception as e:
                logger.error("An error occurred while creating initial source bin", exc_info=e)
            
            try:
                logger.info("Creating Pgie \n ")
                pgie1 = Gst.ElementFactory.make("nvinfer", "primary-inference")
                if not pgie1:
                    sys.stderr.write(" Unable to create pgie \n")
            except Exception as e:
                logger.error("An error occurred while creating pgie1", exc_info=e)
                
            try:
                logger.info("Creating Pgie2 \n ")
                pgie2 = Gst.ElementFactory.make("nvinfer", "primary-inference-2")
                if not pgie2:
                    sys.stderr.write(" Unable to create pgie \n")
            except Exception as e:
                logger.error("An error occurred while creating pgie2", exc_info=e)
            
            try:
                logger.info("Creating sgie1\n ")
                sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-inference-1")
                if not sgie1:
                    sys.stderr.write(" Unable to create sgie1 \n")
            except Exception as e:
                logger.error("An error occurred while creating sgie1", exc_info=e)
                
            try:
                logger.info("Creating sgie2 \n ")
                sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-inference-2")
                if not sgie2:
                    sys.stderr.write(" Unable to create sgie2 \n")    
            except Exception as e:
                logger.error("An error occurred while creating sgie2", exc_info=e)
                
            try:
                logger.info("Creating Nvtracker \n ")
                tracker = Gst.ElementFactory.make("nvtracker","tracker")
                if not tracker:
                    sys.stderr.write(" Unable to create tracker \n")
                pipeline.add(tracker)
            except Exception as e:
                logger.error("An error occurred while creating tracker", exc_info=e)
                
            try:
                logger.info("Creating NvstreamDemux \n ")
                demux = Gst.ElementFactory.make("nvstreamdemux","demux")
                if not demux:
                    sys.stderr.write(" Unable to create demux \n")
            except Exception as e:
                logger.error("An error occurred while creating demux", exc_info=e)
                
            try:
                logger.info("Creating nvvidconv \n ")
                nvvidconv = Gst.ElementFactory.make("nvvideoconvert","nvvidconv")
                if not nvvidconv:
                    sys.stderr.write(" Unable to create nvvideoconvert \n")
            except Exception as e:
                logger.error("An error occurred while creating nvvidconv", exc_info=e)
                
            try:
                logger.info("Creating capsfilter \n")
                capsfilter0 = Gst.ElementFactory.make("capsfilter", "capsfilter0")
                if not capsfilter0:
                    sys.stderr.write(" Unable to create capsfilter0 \n")
                caps0 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                capsfilter0.set_property("caps", caps0)
                
                capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
                if not capsfilter:
                    sys.stderr.write(" Unable to create capsfilter0 \n")
                caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=640, height=640")
                capsfilter.set_property("caps", caps)
            except Exception as e:
                logger.error("An error occurred while creating capsfilters", exc_info=e)
                
            try:
                logger.info("Creating queues \n")
                queue = Gst.ElementFactory.make("queue", "queue_0")
                queue1 = Gst.ElementFactory.make("queue", "queue_1")
                queue2 = Gst.ElementFactory.make("queue", "queue_2")
                queue3 = Gst.ElementFactory.make("queue", "queue_3")
                queue4 = Gst.ElementFactory.make("queue", "queue_4")
            
                queue.set_property("leaky", leaky)
                queue1.set_property("leaky", leaky)
                queue2.set_property("leaky", leaky)
                queue3.set_property("leaky", leaky)
                queue4.set_property("leaky", leaky)
                queue.set_property("max-size-buffers", max_size_buffers)
                queue.set_property("max-size-bytes", max_size_bytes)
                queue.set_property("max-size-time", max_size_time)
            except Exception as e:
                logger.error("An error occurred while creating and setting properties of queues", exc_info=e)
                
            try:
                # streammux.set_property('config-file-path', 'mux_config.txt')
                # streammux.set_property('batch-size', number_sources)
                streammux.set_property("max-latency", max_latency)
                streammux.set_property('sync-inputs', sync_inputs)
                streammux.set_property('width', width)
                streammux.set_property('height', height)
                streammux.set_property('batch-size', number_sources)
                # streammux.set_property('buffer-pool-size',10)
                streammux.set_property('batched-push-timeout', batched_push_timeout)
                streammux.set_property('live-source', live_source)
            except Exception as e:
                logger.error("An error occurred while setting properties of streammux", exc_info=e)
                
            try:
                config = configparser.ConfigParser()
                config.read('./data/dstest2_tracker_config.txt')
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
                    if not is_aarch64():
                        if key == 'enable-batch-process' :
                            tracker_enable_batch_process = config.getint('tracker', key)
                            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
                        if key == 'enable-past-frame' :
                            tracker_enable_past_frame = config.getint('tracker', key)
                            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
            except Exception as e:
                logger.error("An error occurred while configuring properties of tracker", exc_info=e)
                
            try:
                pgie1.set_property('config-file-path', pgie1_path)
                pgie1_batch_size=pgie1.get_property("batch-size")
                logger.info("PGIE1 BATCH SIZE: %d", pgie1_batch_size)
                # if(pgie1_batch_size != number_sources):
                #     logger.info("WARNING: Overriding infer-config batch-size",pgie1_batch_size," with number of sources ", number_sources," \n")
                # pgie1.set_property("batch-size", number_sources)

                sgie1.set_property('config-file-path', sgie1_path)
                sgie1_batch_size=sgie1.get_property("batch-size")
                logger.info("SGIE1 BATCH SIZE: %d", sgie1_batch_size)
                # if(sgie1_batch_size != number_sources):
                #     logger.info("WARNING: Overriding infer-config batch-size",sgie1_batch_size," with number of sources ", number_sources," \n")
                # sgie1.set_property("batch-size", number_sources)

                sgie2.set_property('config-file-path', sgie2_path)
                sgie2_batch_size=sgie2.get_property("batch-size")
                logger.info("SGIE2 BATCH SIZE: %d", sgie2_batch_size)
                # if(sgie2_batch_size != number_sources):
                #     logger.info("WARNING: Overriding infer-config batch-size",sgie2_batch_size," with number of sources ", number_sources," \n")
                # sgie2.set_property("batch-size", number_sources)

                pgie2.set_property('config-file-path', pgie2_path)
                pgie2_batch_size=pgie2.get_property("batch-size")
                logger.info("PGIE2 BATCH SIZE: %d", pgie2_batch_size)
                # if(pgie2_batch_size != number_sources):
                #     logger.info("WARNING: Overriding infer-config batch-size",pgie2_batch_size," with number of sources ", number_sources," \n")
                # pgie2.set_property("batch-size", number_sources)
            except Exception as e:
                logger.error("An error occurred while configuring properties of pgies and sgies", exc_info=e)
                
            try:
                if not is_aarch64():
                    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
                    streammux.set_property("nvbuf-memory-type", mem_type)
                    nvvidconv.set_property("nvbuf-memory-type", mem_type)
            except Exception as e:
                logger.error("An error occurred while configuring memory properties", exc_info=e)
            
            try:
                #adding elements to the pipeline
                logger.info("Adding elements to Pipeline \n")
                pipeline.add(pgie1)
                pipeline.add(pgie2)
                pipeline.add(sgie1)
                pipeline.add(sgie2)
                pipeline.add(queue)
                pipeline.add(queue1)
                pipeline.add(queue2)
                pipeline.add(queue3)
                pipeline.add(queue4)
                pipeline.add(nvvidconv)
                pipeline.add(capsfilter)
                pipeline.add(capsfilter0)
                pipeline.add(demux)
            except Exception as e:
                logger.error("An error occurred while adding elements to Pipeline", exc_info=e)
            
            try:
                #linking all the elements
                logger.info("Linking elements in the Pipeline \n")
                streammux.link(queue)
                queue.link(nvvidconv)
                nvvidconv.link(capsfilter)
                capsfilter.link(queue1)
                queue1.link(pgie1)
                pgie1.link(tracker)
                tracker.link(queue2)
                queue2.link(sgie1)
                sgie1.link(queue3)
                queue3.link(sgie2)
                sgie2.link(queue4)
                queue4.link(pgie2)
                pgie2.link(demux)
            except Exception as e:
                logger.error("An error occurred while linking elements in the Pipeline", exc_info=e)
            
            # try:
            #     for i in range(number_sources):
            #         logger.info("Creating sink %d\n",i)
                    
            #         # Make the encoder
            #         logger.info("Creating H264 Encoder %d\n",i)
            #         encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{i}")
            #         if not encoder:
            #             sys.stderr.write(" Unable to create encoder")
            #         encoder.set_property("bitrate", rtsp_bitrate)
            #         encoder.set_property("control-rate", control_rate)
            #         encoder.set_property("iframeinterval", iframeinterval)
            #         if is_aarch64():
            #             encoder.set_property("insert-sps-pps", insert_sps_pps)
            #             encoder.set_property("vbv-size", vbv_size)
            #             encoder.set_property("maxperf-enable", maxperf_enable)
            #             encoder.set_property("idrinterval", idrinterval)
            #             encoder.set_property("preset-level", preset_level)
            #         else:
            #             encoder.set_property("preset-id", preset_level)
                    
            #         # Make the payload-encode video into RTP packets
            #         logger.info("Creating RTP H264 Payloader %d\n",i)
            #         rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay_{i}")
            #         if not rtppay:
            #             sys.stderr.write(" Unable to create RTP H264 Payloader")
            #         rtppay.set_property("mtu", 500)
                    
            #         logger.info("Creating UDP Sink %d\n",i)
            #         sink = Gst.ElementFactory.make("udpsink", f"udpsink_{i}")
            #         if not sink:
            #             sys.stderr.write(" Unable to create udpsink")
            #         udp_port = UDP_PORT+i
            #         sink.set_property("host", "224.224.255.255")
            #         sink.set_property("port", udp_port)
            #         # sink.set_property("buffer-size", 4000000)
            #         sink.set_property("async", False)
            #         sink.set_property("qos", 0)
            #         sink.set_property("sync", 0)
                    
            #         # creating queue
            #         logger.info("Creating Sink Queues %d\n",i)
            #         queue_0 = Gst.ElementFactory.make("queue", f"queue_1_{i}")
            #         pipeline.add(queue_0)
            #         queue_1 = Gst.ElementFactory.make("queue", f"queue_2_{i}")
            #         pipeline.add(queue_1)
            #         queue_2 = Gst.ElementFactory.make("queue", f"queue_3_{i}")
            #         pipeline.add(queue_2)
            #         queue_3 = Gst.ElementFactory.make("queue", f"queue_4_{i}")
            #         pipeline.add(queue_3)

            #         # creating nvvidconv
            #         logger.info("Creating Sink videoconvert %d\n",i)
            #         nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvc_{i}")
            #         if not nvvideoconvert:
            #             sys.stderr.write(" Unable to create Sink nvvideoconvert")
            #         pipeline.add(nvvideoconvert)
                    
            #         logger.info("Creating Sink capsfilter %d\n",i)
            #         capsfilter_osd = Gst.ElementFactory.make("capsfilter", f"caps_osd_{i}")
            #         pipeline.add(capsfilter_osd)
            #         if not capsfilter_osd:
            #             sys.stderr.write(" Unable to create Sink capsfilter")
            #         caps_osd = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1280, height=720")
            #         capsfilter_osd.set_property("caps", caps_osd)
                    
            #         # creating nvosd
            #         logger.info("Creating Sink OSD %d\n",i)
            #         nvdsosd = Gst.ElementFactory.make("nvdsosd", f"osd_{i}")
            #         if not nvdsosd:
            #             sys.stderr.write(" Unable to create Sink OSD")
            #         pipeline.add(nvdsosd)
            #         nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
            #         nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)
                    
            #         logger.info("Creating Sink nvvidconv %d\n",i)
            #         nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", f"nv_post_ods_{i}")
            #         if not nvvidconv_postosd:
            #             sys.stderr.write(" Unable to create Sink nvvidconv_postosd")
            #         pipeline.add(nvvidconv_postosd)
                    
            #         logger.info("Creating Sink capsfilter %d\n",i)
            #         capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{i}")
            #         if not capsfilter:
            #             sys.stderr.write(" Unable to create Sink capsfilter")
            #         pipeline.add(capsfilter)
            #         caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720, format=I420")
            #         capsfilter.set_property("caps", caps)
                    
            #         if not is_aarch64():
            #             mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            #             nvvideoconvert.set_property("nvbuf-memory-type", mem_type)
            #             nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)
                    
            #         # connect nvstreamdemux -> queue
            #         padname = "src_%u" %i
            #         demuxsrcpad = demux.get_request_pad(padname)
            #         if not demuxsrcpad:
            #             sys.stderr.write("Unable to create demux src pad \n")
            #         queuesinkpad = queue_0.get_static_pad("sink")
            #         if not queuesinkpad:
            #             sys.stderr.write("Unable to create queue sink pad \n")
            #         demuxsrcpad.link(queuesinkpad)
                    
            #         pipeline.add(encoder)
            #         pipeline.add(rtppay)
            #         pipeline.add(sink)
                
            #         queue_0.link(nvvideoconvert)
            #         nvvideoconvert.link(capsfilter_osd)
            #         capsfilter_osd.link(nvdsosd)
            #         nvdsosd.link(queue_1)
            #         queue_1.link(nvvidconv_postosd)
            #         nvvidconv_postosd.link(queue_2)
            #         queue_2.link(capsfilter)
            #         capsfilter.link(queue_3)
            #         queue_3.link(encoder)
            #         encoder.link(rtppay)
            #         rtppay.link(sink)
                    
            #         factory = GstRtspServer.RTSPMediaFactory.new()
            #         factory.set_launch(
            #         '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
            #         % (udp_port, codec))
                    
            #         factory.set_shared(True)
            #         server.get_mount_points().add_factory(f"/ds-test-{i}", factory)
        
            #         logger.info(
            #         f"\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test-{i} ***\n\n"
            #         % rtsp_port)                    
            # except Exception as e:
            #     logger.error("An error occurred while creating sink pipeline", exc_info=e)
            
            try:
                # create an event loop and feed gstreamer bus mesages to it
                loop = GLib.MainLoop()
                bus = pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect ("message", bus_call, loop)
            except Exception as e:
                logger.error("An error occurred while creating event loop and feeding gstreamer bus mesages to MainLoop", exc_info=e)
            
            try:
                tracker_src_pad=pgie2.get_static_pad("src")
                if not tracker_src_pad:
                    sys.stderr.write(" Unable to get src pad \n")
                else:
                    tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, args)
            except Exception as e:
                logger.error("An error occurred while adding probe to tracker src pad", exc_info=e)
                
            try:
                logger.info("Starting pipeline \n")
                # start play back and listed to events		
                pipeline.set_state(Gst.State.PLAYING)
            except Exception as e:
                logger.error("Unable to set pipeline to play state", exc_info=e)
            try:
                loop.run()
            except:
                pass
        else:
            logger.info("Not enough devices, Exiting app!!\n")
            pipeline.set_state(Gst.State.NULL)
            
    except Exception as e:
        logger.error("An error occurred in number of devices. Exiting the deepstream app.", exc_info=e)
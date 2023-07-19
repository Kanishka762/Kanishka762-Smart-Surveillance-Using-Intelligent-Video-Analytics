
#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


import argparse
import sys
sys.path.append('../')

import os
from os.path import join, dirname
import time
import threading
import random
import string
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from datetime import datetime
import datetime
import shutil

# from common.FPS import GETFPS
# from sparkplug_b import *
import pyds
import pytz
import datetime
import numpy as np
import cv2
import asyncio
import threading 

from json_structure import frame_2_dict
from testing_tenant import filter_devices
from dotenv import load_dotenv
from gif_creation import gif_build
from db_fetch_members import fetch_db_mem
from facedatainsert_lmdb import add_member_to_lmdb
from lmdb_list_gen import attendance_lmdb_known, attendance_lmdb_unknown

path = os.getcwd()

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

device = os.getenv("device")
tenant_name = os.getenv("tenant_name")

config_path = path + f"/models_deepstream/{tenant_name}/{device}/config.txt"


hls_path = "./Hls_output"
if os.path.exists(hls_path) is False:
    os.mkdir(hls_path)
 
# create a sub-directory to hold the segment and manifest files



timezone = pytz.timezone('Asia/Kolkata')   #assign timezone


dev_id_dict = {}
gif_dict = {}

detect_data = []
MAX_DISPLAY_LEN=64

if tenant_name == 'YOLOv8' : 
    PGIE_CLASS_ID_MALE = 0
    PGIE_CLASS_ID_FEMALE = 1
    PGIE_CLASS_ID_FIRE = 2
    PGIE_CLASS_ID_SMOKE = 3
    PGIE_CLASS_ID_GUN = 4
    PGIE_CLASS_ID_KNIFE = 5
    pgie_classes_str= [ "Male","Female","Fire","Smoke","Gun","Knife"]
elif tenant_name == "YOLOv5" :
    PGIE_CLASS_ID_CARRYING = 0
    PGIE_CLASS_ID_THROWING = 1
    PGIE_CLASS_ID_SITTING = 2
    PGIE_CLASS_ID_STANDING = 3
    PGIE_CLASS_ID_WALKING = 4
    PGIE_CLASS_ID_SITTING_ON_THE_BOX = 5
    PGIE_CLASS_ID_STANDING_ON_THE_BOX = 6
    pgie_classes_str= [ "carrying" , "standing", "sitting", "walking","throwing","sitting on the box","standing on the box" ]
elif tenant_name =='Bagdogra' :
    PGIE_CLASS_ID_PERSON = 0
    PGIE_CLASS_ID_VEHICLE = 1
    PGIE_CLASS_ID_ELEPHANT = 2
    pgie_classes_str= [ "Person", "Vehicle","Elephant"]
# PGIE_CLASS_ID_STANDING_ON_THE_BOX = 6


past_tracking_meta=[0]

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1


def load_lmdb_list():
    known_whitelist_faces1, known_whitelist_id1 = attendance_lmdb_known()
    known_blacklist_faces1, known_blacklist_id1 = attendance_lmdb_unknown()
    
    global known_whitelist_faces
    known_whitelist_faces = known_whitelist_faces1

    global known_whitelist_id
    known_whitelist_id = known_whitelist_id1
    
    global known_blacklist_faces
    known_blacklist_faces = known_blacklist_faces1

    global known_blacklist_id
    known_blacklist_id = known_blacklist_id1

def load_lmdb_fst(mem_data):
    i = 0
    for each in mem_data:
        i = i+1
        add_member_to_lmdb(each)
        print("inserting ",each)

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name + ',C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255, 0), 2)
    return image

def crop_object(image, obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]

    crop_img = image[top:top+height, left:left+width]
	
    return crop_img

def tracker_src_pad_buffer_probe(pad,info,u_data):
    global gif_dict
    
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_MALE : 0,
        PGIE_CLASS_ID_FEMALE : 0,
        PGIE_CLASS_ID_FIRE : 0,
        PGIE_CLASS_ID_SMOKE : 0,
        PGIE_CLASS_ID_GUN : 0,
        PGIE_CLASS_ID_KNIFE : 0

    }
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    # buf_surface = pyds.get_nvds_buf_surface(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data) #new frame
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            n_frame = cv2.cvtColor(n_frame, cv2.COLOR_BGR2RGB)
            dt = datetime.datetime.now(timezone)
            
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        frame_copy = np.array(n_frame, copy = True, order = 'C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2RGB)
        
        n_frame_copy = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2RGB)
        # cv2.imwrite(f'{frame_number}.jpg',frame_copy)
        
        camera_id = frame_meta.pad_index
        source_Id = frame_meta.source_id
        
        for key, value in dev_id_dict.items():
            device_id = value['deviceId']
            if device_id not in gif_dict:
                gif_dict[device_id] = []
            
        # t1 = threading.Thread(target = gif_build, args = (n_frame_copy, dev_id_dict[camera_id], gif_dict,))
        # t1.start()
        
                
        asyncio.run(gif_build(n_frame_copy, dev_id_dict[camera_id], gif_dict))
        
        num_detect = frame_meta.num_obj_meta
        device_timestamp = datetime.datetime.now(timezone)
        frame_dict = {
        'frame_number': frame_number ,
        'total_detect' : num_detect,
        'camera_id' : camera_id,
        'frame_timestamp' : device_timestamp,
        'objects': []  # List to hold object dictionaries
    }
        l_obj=frame_meta.obj_meta_list
        # print(frame_meta)
        # print(dev_id_dict[camera_id])
        n_frame_bbox = None
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data) #new obj
                obj_counter[obj_meta.class_id] += 1

                confidence_score = obj_meta.confidence
                
                detect_type = obj_meta.obj_label
                
                bbox = obj_meta.tracker_bbox_info 
                
                rect_params = obj_meta.rect_params
                left = rect_params.left
                top = rect_params.top
                width = rect_params.width
                height = rect_params.height
 
                n_frame_bbox = None
                n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence)
                # cv2.imwrite(f"/home/agx123/DS_pipeline_new/frame_bbox/{frame_number}.jpg",n_frame_bbox)
                # # convert python array into numpy array format in the copy mode.
                # frame_bbox = np.array(n_frame_bbox, copy=True, order='C')
                # # convert the array into cv2 default color format
                # frame_bbox = cv2.cvtColor(frame_bbox, cv2.COLOR_RGBA2BGRA)
                
                n_frame_crop = crop_object(frame_copy, obj_meta)
                # # convert python array into numpy array format in the copy mode.
                # frame_crop_copy = np.array(n_frame_crop, copy=True, order='C')
                # # convert the array into cv2 default color format
                frame_crop_copy = cv2.cvtColor(n_frame_crop, cv2.COLOR_RGBA2BGRA)
                # cv2.imwrite(f"/home/agx123/DS_pipeline_new/frame_crops/{frame_number}.jpg",frame_crop_copy)
                
                obj_id  =  obj_meta.object_id
                obj_dict =  {
                'detect_type' : detect_type,
                'confidence_score': confidence_score,
                'obj_id' : obj_id,
                'bbox_left' : left,
                'bbox_top' : top,
                'bbox_right' : width,
                'bbox_bottom' : height,
                'timestamp' :  dt.strftime("%H:%M:%S %d/%m/%Y"),
                'crop' : cv2.cvtColor(frame_crop_copy, cv2.COLOR_BGR2RGB)
                # 'dirs' : x,
                # 'tracker' : y

                }
                frame_dict['objects'].append(obj_dict)
                detect_data.append(frame_dict)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Male={} Female={} Knife ={} Gun={}".format(frame_number, obj_counter[PGIE_CLASS_ID_MALE], obj_counter[PGIE_CLASS_ID_FEMALE], obj_counter[PGIE_CLASS_ID_KNIFE], obj_counter[PGIE_CLASS_ID_GUN])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next  
            if n_frame_bbox is not None:
                frame_dict['np_arr'] = n_frame_bbox   
            else:
                frame_dict['np_arr'] = frame_copy
            datainfo = [known_whitelist_faces, known_blacklist_faces, known_whitelist_id, known_blacklist_id]       
                   
            frame_2_dict(frame_dict,dev_id_dict,datainfo)
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK	

 ######################################################################


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)



def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    
        # use nvurisrcbin to enable file-loop
    uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
    uri_decode_bin.set_property("rtsp-reconnect-interval", 50)
    # uri_decode_bin.set_property("file-loop", "true")
    # uri_decode_bin.set_property("cudadec-memtype", 0)
    # else:
    #     uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    # if not uri_decode_bin:
    #     sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

 ######################################################################

def main(args):
    print(args)
    global dev_id_dict
    # Check input arguments
    past_tracking_meta[0]=1
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    # for i in range(0,len(args)-1):
    #     fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-1
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
        uri_name = args[i+1]['rtsp']
        # print("+++++++++++++++")
        # print(uri_name)
        # print("+++++++++++++++")

        # dev_id = args[i+1]['deviceId']
        dev_id_dict[i] = args[i+1]
        print(dev_id_dict)

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

    print(dev_id_dict)
    print("creating nvvideoconvert")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert","nvvidconv")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvideoconvert \n")

    print("creating tee")
    tee = Gst.ElementFactory.make("tee","tee1")
    if not tee:
        sys.stderr.write(" Unable to create tee \n")

    print("creating demxuer queue")
    queue_d = Gst.ElementFactory.make("queue","demux")
    if not queue_d:
        sys.stderr.write(" Unable to create tiler queue \n")
    
    print("creating tiler queue")
    queue_t = Gst.ElementFactory.make("queue","tiler")
    if not queue_t:
        sys.stderr.write(" Unable to create demuxer queue \n")

    print("Creating demuxer \n ")
    demux=Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        sys.stderr.write(" Unable to create demuxer \n")

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating Tracker \n")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    
    print("Creating capsfilter \n")

    capsfilter0 = Gst.ElementFactory.make("capsfilter", "capsfilter0")
    if not capsfilter0:
        sys.stderr.write(" Unable to create capsfilter0 \n")

    caps0 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter0.set_property("caps", caps0)

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("creating nvvideoconvert post tiler")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert","nvvidconv1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvideoconvert  post tiler \n")
    
    print("Creating nvosd \n ")
    nvosd_tiler = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
           ###########################################
    if not nvosd_tiler:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd_tiler.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd_tiler.set_property('display-text',OSD_DISPLAY_TEXT)

    print("creating display sink")
    sink = Gst.ElementFactory.make("nv3dsink","sink")
    if not sink:
        sys.stderr.write(" Unable to create sink \n")

    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
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
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
    
    streammux.set_property('gpu-id', 0)
    streammux.set_property('enable-padding', 0)
    # streammux.set_property('nvbuf-memory-type', 0)
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', 5)
    streammux.set_property('batched-push-timeout', 40000)
    pgie.set_property('config-file-path', config_path)
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
    pgie.set_property("batch-size", number_sources)
    

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
     

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter0)
    pipeline.add(demux)
    pipeline.add(tee)
    pipeline.add(tiler)
    pipeline.add(queue_d)
    pipeline.add(queue_t)
    pipeline.add(nvvidconv1)
    pipeline.add(sink)
    pipeline.add(nvosd_tiler)


    

    print("Linking elements in the Pipeline \n")
    streammux.link(nvvidconv)
    nvvidconv.link(capsfilter0)
    capsfilter0.link(pgie)
    pgie.link(tracker)
    tracker.link(tee)
    tee.link(queue_d)
    tee.link(queue_t)
    queue_d.link(demux)
    queue_t.link(tiler)
    tiler.link(nvvidconv1)
    nvvidconv1.link(nvosd_tiler)
    nvosd_tiler.link(sink)

    for i in range(number_sources):
        print("Creating sink ",i," \n ")

        DDNS = args[i+1]['ddns']
        if DDNS is None or " ":
            DDNS = "localhost:8080"

        

        video_info = hls_path + '/' + dev_id_dict[i]['deviceId']
        if not os.path.exists(video_info):
            os.makedirs(video_info, exist_ok=True)
        
        sink = Gst.ElementFactory.make("hlssink", f"sink_{i}")
        pipeline.add(sink)
        devid = dev_id_dict[i]['deviceId']
        sink.set_property('playlist-root', f'http://{DDNS}/demo/{devid}/') # Location of the playlist to write
        
        sink.set_property('playlist-location', f'{video_info}/{devid}.m3u8') # Location where .m3u8 playlist file will be stored
        sink.set_property('location',  f'{video_info}/segment.%01d.ts')  # Location whee .ts segmentrs will be stored
        sink.set_property('target-duration', 3) # The target duration in seconds of a segment/file. (0 - disabled, useful
        sink.set_property('playlist-length', 2) # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
        sink.set_property('max-files', 6) # Maximum number of files to keep on disk. Once the maximum is reached,old files start to be deleted to make room for new ones.
        
        # creating queue
        queue_0 = Gst.ElementFactory.make("queue", f"queue_1_{i}")
        pipeline.add(queue_0)

        queue_1 = Gst.ElementFactory.make("queue", f"queue_2_{i}")
        pipeline.add(queue_1)
        queue_2 = Gst.ElementFactory.make("queue", f"queue_3_{i}")
        pipeline.add(queue_2)
        queue_3 = Gst.ElementFactory.make("queue", f"queue_4_{i}")
        pipeline.add(queue_3)
        queue_4 = Gst.ElementFactory.make("queue", f"queue_5_{i}")
        pipeline.add(queue_4)
        queue_5 = Gst.ElementFactory.make("queue", f"queue_6_{i}")
        pipeline.add(queue_5)

        parser = Gst.ElementFactory.make("h264parse", f"parse_{i}")
        pipeline.add(parser)


        # creating nvvidconv
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvc_{i}")
        pipeline.add(nvvideoconvert)
        # creating nvosd

        nvdsosd = Gst.ElementFactory.make("nvdsosd", f"osd_{i}")
        pipeline.add(nvdsosd)
        nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", f"nv_post_ods{i}")
        pipeline.add(nvvidconv_postosd)

        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{i}")
        pipeline.add(capsfilter)
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720")
        capsfilter.set_property("caps", caps)
        
        
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{i}") # nvv4l2h264enc
        pipeline.add(encoder)
        encoder.set_property("bitrate", 180000)
        encoder.set_property("preset-level", "FastPreset")
        container = Gst.ElementFactory.make("mpegtsmux", f"mux_{i}")
        pipeline.add(container)
        if not is_aarch64():
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            nvvideoconvert.set_property("nvbuf-memory-type", mem_type)
            nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)
        # connect nvstreamdemux -> queue
        padname = "src_%u" %i
        demuxsrcpad = demux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")
        queuesinkpad = queue_0.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)
        
        queue_0.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)
        nvdsosd.link(queue_1)
        queue_1.link(nvvidconv_postosd)
        nvvidconv_postosd.link(queue_2)
        queue_2.link(capsfilter)
        capsfilter.link(queue_3)
        queue_3.link(encoder)
        encoder.link(parser)
        parser.link(queue_4)
        queue_4.link(container)
        container.link(queue_5)
        queue_5.link(sink)


    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    
    tracker_src_pad=tracker.get_static_pad("src")
    if not tracker_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)


    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

######################################################################
    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
        
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    # sys.exit(main(sys.argv))

    if os.path.exists("./Hls_output"):
        # Delete Folder code
        shutil.rmtree("./Hls_output")
    
    if os.path.exists("./Gif_output"):
        # Delete Folder code
        shutil.rmtree("./Gif_output")
    
    load_lmdb_list()
    print("removed lmdb contents")
    mem_data = fetch_db_mem()
    print(mem_data)
    load_lmdb_fst(mem_data)
    load_lmdb_list()



    device_det = filter_devices()
    print(device_det)
    dev_details = [""]
    for chunk in device_det:
        device_dict = {}
        device_dict["deviceId"] = chunk[0]
        device_dict["tenantId"] = chunk[1]
        device_dict["urn"] = chunk[2]
        device_dict["ddns"] = chunk[3]
        device_dict["ip"] = chunk[4]
        device_dict["port"] = chunk[5]
        device_dict["videoEncodingInformation"] = chunk[6]
        # device_dict["videoEncodingInformation"] = 'MP4'
        device_dict["username"] = chunk[7]
        #device_dict["rtsp"] = chunk[8]
        device_dict["rtsp"] = "/home/agx123/face_recog_test.mp4"
        # device_dict["rtsp"] = dev_list[index]
        device_dict["password"] = chunk[9]
        device_dict["subscriptions"] = chunk[10]
        device_dict["lat"] = chunk[11]
        device_dict["long"] = chunk[12]
        dev_details.append(device_dict)
    main(dev_details)
    # # device_dict = {}




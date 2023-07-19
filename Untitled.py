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
import os
import sys
sys.path.append('../')


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
# from common.FPS import GETFPS
# from sparkplug_b import *
import pyds
import pytz
import datetime
import numpy as np
import cv2
# os.environ["GST_DEBUG"] = "3"

timezone = pytz.timezone('Asia/Kolkata')   #assign timezone

# create a directory to hold the segment and manifest files
# hls_path = "./Hls_output"
# if os.path.exists(hls_path) is False:
#     os.mkdir(hls_path)
    


detect_data = []
MAX_DISPLAY_LEN=64

PGIE_CLASS_ID_MALE = 0
PGIE_CLASS_ID_FEMALE = 1
PGIE_CLASS_ID_FIRE = 2
PGIE_CLASS_ID_SMOKE = 3
PGIE_CLASS_ID_GUN = 4
PGIE_CLASS_ID_KNIFE = 5
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
pgie_classes_str= [ "Male","Female","Fire","Smoke","Gun","Knife"]
    

def tracker_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    # obj_counter = {
    #     PGIE_CLASS_ID_CARRYING : 0,
    #     PGIE_CLASS_ID_THROWING : 0,
    #     PGIE_CLASS_ID_SITTING : 0,
    #     PGIE_CLASS_ID_STANDING : 0,
    #     PGIE_CLASS_ID_WALKING : 0,
    #     PGIE_CLASS_ID_SITTING_ON_THE_BOX : 0,
    #     PGIE_CLASS_ID_STANDING_ON_THE_BOX : 0
        
       

    # }
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
            dt = datetime.datetime.now(timezone)
        
            
        except StopIteration:
            break
        
        
        frame_number = frame_meta.frame_num
        frame_copy = np.array(n_frame, copy = True, order = 'C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2RGB)
        # cv2.imwrite(f'/home/agx123/DS_pipeline_new/Hls_output_1/{frame_number}.jpg',frame_copy)
        
        camera_id = frame_meta.pad_index
        source_Id = frame_meta.source_id

        
        num_detect = frame_meta.num_obj_meta
        frame_dict = {
        'frame_number': frame_number ,
        'total_detect' : num_detect,
        'source_id' : source_Id,
        'camera_id' : camera_id,
        'np_arr' : frame_copy,
        'objects': []  # List to hold object dictionaries
    }
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data) #new obj
                # obj_counter[obj_meta.class_id] += 1

                confidence_score = obj_meta.confidence
                
                detect_type = obj_meta.obj_label
                
                bbox = obj_meta.tracker_bbox_info 
                rect_params = obj_meta.rect_params
                left = rect_params.left
                top = rect_params.top
                width = rect_params.width
                height = rect_params.height
                
                obj_id  =  obj_meta.object_id
                obj_dict =  {
                'detect_type' : detect_type,
                'confidence_score': confidence_score,
                'obj_id' : obj_id,
                'bbox_left' : left,
                'bbox_top' : top,
                'bbox_right' : width,
                'bbox_bottom' : height,
                'timestamp' :  dt.strftime("%H:%M:%S %d/%m/%Y")
                # 'dirs' : x,
                # 'tracker' : y

                }
                frame_dict['objects'].append(obj_dict)
                detect_data.append(frame_dict)
            except StopIteration:
                break
            # obj_counter[obj_meta.class_id] += 1
            
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
        # py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

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
            print(frame_dict)
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK	

 ######################################################################

# def make_element(element_name, i):
#     """
#     Creates a Gstreamer element with unique name
#     Unique name is created by adding element type and index e.g. `element_name-i`
#     Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
#     :param element_name: The name of the element to create
#     :param i: the index of the element in the pipeline
#     :return: A Gst.Element object
#     """
#     element = Gst.ElementFactory.make(element_name, element_name)
#     if not element:
#         sys.stderr.write(" Unable to create {0}".format(element_name))
#     element.set_property("name", "{0}-{1}".format(element_name, str(i)))
#     return element


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
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
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
    # queue1=Gst.ElementFactory.make("queue","queue1")
    # queue2=Gst.ElementFactory.make("queue","queue2")
    # queue3=Gst.ElementFactory.make("queue","queue3")
    # queue4=Gst.ElementFactory.make("queue","queue4")
    # queue5=Gst.ElementFactory.make("queue","queue5")
    # pipeline.add(queue1)
    # pipeline.add(queue2)
    # pipeline.add(queue3)
    # pipeline.add(queue4)
    # pipeline.add(queue5)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating Tracker \n")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    # print("Creating tiler \n ")
    # tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    # if not tiler:
    #     sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")  
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")  
    # if not nvvidconv:
    #     sys.stderr.write(" Unable to create nvvidconv \n")
    # print("Creating nvosd \n ")
    # nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

           ###########################################
    # if not nvosd:
    #     sys.stderr.write(" Unable to create nvosd \n")
    # nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    # nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    # #if(is_aarch64()):
    #     #print("Creating transform \n ")
    #     #transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    #     #if not transform:
    #         #sys.stderr.write(" Unable to create transform \n")

    
    # nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    # if not nvvidconv_postosd:
    #     sys.stderr.write(" Unable to create nvvidconv_postosd \n")
        
    
    # print("Creating capsfilter \n")

    capsfilter0 = Gst.ElementFactory.make("capsfilter", "capsfilter0")
    if not capsfilter0:
        sys.stderr.write(" Unable to create capsfilter0 \n")

    caps0 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter0.set_property("caps", caps0)

    # capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    # if not capsfilter:
    #     sys.stderr.write(" Unable to create capsfilter \n")

    # caps = Gst.Caps.from_string("video/x-raw, width=1280, height=720")
    # capsfilter.set_property("caps", caps)

    # print("Creating Encoder \n")
    # encoder = Gst.ElementFactory.make("x264enc", "encoder")
    # if not encoder:
    #     sys.stderr.write(" Unable to create encoder \n")

  

    # print("Creating Code Parser \n")
    # codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    # if not codeparser:
    #     sys.stderr.write(" Unable to create code parser \n")

    # print("Creating Container \n")
    # container = Gst.ElementFactory.make("mpegtsmux", "mux")
    # if not container:
    #     sys.stderr.write(" Unable to create code parser \n")

    print("Creating demuxer \n ")
    demux=Gst.ElementFactory.make("nvstreamdemux", "demuxer")
    if not demux:
        sys.stderr.write(" Unable to create demuxer \n")
    
        
        

    # Make the UDP sink
    # updsink_port_num = 5400
    # sink = Gst.ElementFactory.make("filesink", "udpsink")
    # if not sink:
    #     sys.stderr.write(" Unable to create hlssink")

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

    
    
    # sink.set_property('host', '224.224.255.255')
    # sink.set_property('port', updsink_port_num)
    # sink.set_property('async', False)
    # sink.set_property('sync', 0)

    # sink.set_property("location", "./out_v5.mp4")
    # sink.set_property("sync", 1)
    # sink.set_property("async", 0)
    
    streammux.set_property('gpu-id', 0)
    streammux.set_property('enable-padding', 0)
    # streammux.set_property('nvbuf-memory-type', 0)
    streammux.set_property('width', 640)
    streammux.set_property('height', 480)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    pgie.set_property('config-file-path', "/home/agx123/DS_pipeline_new/models_deepstream/TCI_YOLOv8/jetson/config.txt")
    #pgie_batch_size=pgie.get_property("batch-size")
    #if(pgie_batch_size != number_sources):
        #print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
    # pgie.set_property("batch-size", 2)
    
  

    # tiler_rows=int(math.sqrt(number_sources))
    # tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    # tiler.set_property("rows",tiler_rows)
    # tiler.set_property("columns",tiler_columns)
    # tiler.set_property("width", TILED_OUTPUT_WIDTH)
    # tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    # if not is_aarch64():
        # mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    # streammux.set_property("nvbuf-memory-type", 0)
    # nvvidconv.set_property("nvbuf-memory-type", 0)
    
    # demux.set_property("nvbuf-memory-type", mem_type)
    # sink.set_property("qos",0)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(demux)
    pipeline.add(capsfilter0)
    
    

    print("Linking elements in the Pipeline \n")
    streammux.link(nvvidconv)
    nvvidconv.link(capsfilter0)
    capsfilter0.link(pgie)
    pgie.link(tracker)
    tracker.link(demux)
    

    
    for i in range(number_sources):
        print("Creating sink ",i," \n ")

        # video_info = hls_path + '/' + str(i)
        # if not os.path.exists(video_info):
        #     os.makedirs(video_info, exist_ok=True)
        
        sink = Gst.ElementFactory.make("hlssink", f"sink_{i}")
        pipeline.add(sink)

        sink.set_property(f'playlist-root', f'http://localhost:8999/Hls_output_1') # Location of the playlist to write
        sink.set_property('playlist-location', f'/home/agx123/DS_pipeline_new/Hls_output_1/playlist_.m3u8') # Location where .m3u8 playlist file will be stored
        sink.set_property('location',  f'/home/agx123/DS_pipeline_new/Hls_output_1/seg_low_bitrate_segmet.%01d.ts')  # Location whee .ts segmentrs will be stored
        sink.set_property('target-duration', 6) # The target duration in seconds of a segment/file. (0 - disabled, useful for management of segment duration by the streaming server)
        sink.set_property('playlist-length', 3) # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
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

        # capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{i}")
        # pipeline.add(capsfilter)
        # caps = Gst.Caps.from_string("video/x-raw, width=1280, height=720")
        # capsfilter.set_property("caps", caps)
        

        
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"enoder_{i}")
        pipeline.add(encoder)
        encoder.set_property("bitrate", 1800000)
        # encoder.set_property("speed-preset", "veryfast")
        encoder.set_property("preset-level", "UltraFastPreset")

        container = Gst.ElementFactory.make("mpegtsmux", f"mux_{i}")
        pipeline.add(container)

        # if not is_aarch64():
            # mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        # nvvideoconvert.set_property("nvbuf-memory-type", 0)
        # nvvidconv_postosd.set_property("nvbuf-memory-type", 0)



        # connect nvstreamdemux -> queue
        padname = "src_%u" %i
        demuxsrcpad = demux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad \n")

        queuesinkpad = nvvideoconvert.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad \n")
        demuxsrcpad.link(queuesinkpad)

        # queue.link(nvvideoconvert)
        # queue_0.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)
        nvdsosd.link(queue_1)
        queue_1.link(nvvidconv_postosd)
        nvvidconv_postosd.link(queue_2)
        # queue_2.link(capsfilter)
        # capsfilter.link(queue_3)
        queue_2.link(encoder)
        # encoder.link(parser)
        encoder.link(queue_4)
        queue_4.link(container)
        container.link(queue_5)

        # encoder.link(container)
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
    sys.exit(main(sys.argv))


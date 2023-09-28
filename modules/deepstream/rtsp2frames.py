
from modules.components.load_paths import *
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
from datetime import datetime
import datetime
import pyds
import pytz
# import datetime
import numpy as np
import cv2
import asyncio
from dotenv import load_dotenv
import lmdb
import json
import threading

from modules.gif.gif_creation import gif_build
from modules.db.db_push import gif_push, gst_hls_push
from modules.components.generate_crop import save_one_box
# from modules.data_process.frame_data_process import frame_2_dict
# from modules.face_recognition_pack.lmdb_list_gen import attendance_lmdb_known, attendance_lmdb_unknown

# from modules.face_recognition_pack.lmdb_components import known_whitelist_faces,known_whitelist_id,known_blacklist_faces,known_blacklist_id

import queue

framedata_queue = queue.Queue()

load_dotenv(dotenv_path)

device = os.getenv("device")
tenant_name = os.getenv("tenant_name")
ddns_name = os.getenv("DDNS_NAME")
place = os.getenv("place")
obj_det_labels = ast.literal_eval(os.getenv("obj_det_labels"))
anomaly_objs = ast.literal_eval(os.getenv("anamoly_object"))
classDict = ast.literal_eval(os.getenv("classDict"))
constantIdObjects = ast.literal_eval(os.getenv("constantIdObjects"))
track_type = ast.literal_eval(os.getenv("track_type"))
frame_bbox_color = ast.literal_eval(os.getenv("color"))

timezone = pytz.timezone(f'{place}')  #assign timezone
pgie1_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/gender/config.txt"
pgie2_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/fire/config.txt"
sgie1_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/face_detect/config.txt"
sgie2_path = f"{cwd}/models_deepstream/{tenant_name}/{device}/dangerous_object/config.txt"
frame_path = f"{cwd}/static/frames/"
infer_path = f"{cwd}/static/image/"
crop_path = f"{cwd}/static/crops/"

age_dict = {}
dev_id_dict = {}
gif_dict = {}
detect_data = []
gif_created = {}
cnt = 0
label_dict = {label: i for i, label in enumerate(obj_det_labels)}
PRIMARY_DETECTOR_UID_1 = 1
PRIMARY_DETECTOR_UID_2 = 2
SECONDARY_DETECTOR_UID_1 = 3
SECONDARY_DETECTOR_UID_2 = 4

past_tracking_meta=[0]
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
pgie_classes_str= [ "Male","Female","Fire","Smoke","Gun","Knife"]
BITRATE  = 4000000

db_env = lmdb.open(f'{lmdb_path}/face-detection', max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)

def draw_bounding_boxes(image, obj_meta, confidence, label_str, bbox_color):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    
    obj_name = obj_meta.obj_label
    # image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = frame_bbox_color[bbox_color]
    w_percents = int(width * 0.03) if width > 100 else int(width * 0.05)
    h_percents = int(height * 0.03) if height > 100 else int(height * 0.05)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 2)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 2)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 2)
    lineright_c1 = (left + width, top + h_percents) 
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 2)
    
    # Calculate the coordinates for the background polygon
    text_size, _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_width, text_height = text_size
    bg_left = left - 10
    bg_top = top - 10
    bg_right = bg_left + text_width + 20
    bg_bottom = bg_top + text_height + 20
    
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, label_str, (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)
    return image


def crop_object(image, obj_meta):
    rect_params = obj_meta.rect_params
    x1, y1, x2, y2 = map(int, (rect_params.left, rect_params.top, rect_params.left + rect_params.width, rect_params.top + rect_params.height))
    crop_img = save_one_box([x1, y1, x2, y2], image)
    return cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

def findClassList(subscriptions):
    subscriptions_class_list = [item for sublist in [classDict[each] for each in subscriptions if each in classDict] for item in sublist]
    subscriptions_class_list.extend(iter(obj_det_labels))
    return subscriptions_class_list

#TODO: check new logics are right
def fetch_activity_info(detect_type):
    key = "IdLabelInfo"
    output_dict = {}
    with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
        value = db_txn.get(key.encode())
        if value is None:
            return None
        data = json.loads(value.decode())
        for key, value in data.items():
            memID = value['memberID']
            activities = list(set(value['activity']))

            if memID is None:
                member_track_type = None

            elif memID == '100':
                if memID in track_type:
                    member_track_type = track_type[memID]
            else:
                mem_type = memID[:2]
                if mem_type in track_type:
                    member_track_type = track_type[mem_type]
            sentence = (
                f"{detect_type} {' '.join(activities)}"
                if member_track_type is None
                else f"{member_track_type} {detect_type} {' '.join(activities)}"
            )
            output_dict[key] = sentence

        return output_dict

#TODO: check new logics are right
def fetch_member_info(detect_type):
    key = "trackIdMemIdDict"
    output_dict = {}
    with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
        value = db_txn.get(key.encode())
    if value is None:
        return None
    data = json.loads(value.decode())
    for key, value in data.items():
        memID = (data[key])[-1]

        if memID is None:
            member_track_type = None

        elif memID == '100':
            if memID in track_type:
                member_track_type = track_type[memID]
        else:
            mem_type = memID[:2]
            if mem_type in track_type:
                member_track_type = track_type[mem_type]
        sentence = (
            f"{detect_type}"
            if member_track_type is None
            else f"{member_track_type} {detect_type}"
        )
        output_dict[key] = sentence
    return output_dict

def tracker_src_pad_buffer_probe(pad,info,dev_list):
    
    # print("called callback")
    global gif_dict,cnt

    frame_number=0
    
    #Intiallizing object counter with 0.
    obj_counter = {}
    
    for label in label_dict:
        obj_counter[label_dict[label]] = 0 
    # print(obj_counter)

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
        # cv2.imwrite(f'{frame_path}/{frame_number}.jpg',frame_copy)
        
        camera_id = frame_meta.pad_index
        
        dev_info = dev_list[camera_id]
        subscriptions = dev_info['subscriptions']
        subscriptions_class_list = findClassList(subscriptions)
        for key, value in dev_id_dict.items():
            device_id = value['deviceId']
            if device_id not in gif_dict:
                gif_dict[device_id] = []
            if device_id not in gif_created:
                gif_created[device_id] = False
                
        if 'Bagdogra' not in subscriptions:
            threading.Thread(target = gif_build,args = (n_frame_copy, dev_id_dict[camera_id], gif_dict, gif_created,)).start()
                
        num_detect = frame_meta.num_obj_meta
        # print(datetime.datetime.now(timezone))
        device_timestamp = datetime.datetime.now(timezone)
        frame_dict = {
        'frame_number': frame_number ,
        'total_detect' : num_detect,
        'camera_id' : camera_id,
        'frame_timestamp' : device_timestamp,
        'objects': []  # List to hold object dictionaries
    }
        # print(frame_dict)
        l_obj=frame_meta.obj_meta_list
        # print(dev_id_dict[camera_id])
        n_frame_bbox = None
        output_lbl = None
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data) #new obj
                obj_counter[obj_meta.class_id] += 1
                confidence_score = obj_meta.confidence
                detect_type = obj_meta.obj_label
                rect_params = obj_meta.rect_params
                text_params = obj_meta.text_params
                left = rect_params.left
                top = rect_params.top
                width = rect_params.width
                height = rect_params.height
                # print('detect_type',detect_type)
                if(obj_meta.unique_component_id == PRIMARY_DETECTOR_UID_1):
                    bbox_color = "green"
                    rect_params.has_bg_color = 1
                    rect_params.border_color.red = 0.0
                    rect_params.border_color.green = 1.0
                    rect_params.border_color.blue = 0.0
                    rect_params.border_color.alpha = 1.0
                    rect_params.bg_color.red = 0.0
                    rect_params.bg_color.green = 1.0
                    rect_params.bg_color.blue = 0.0
                    rect_params.bg_color.alpha = 0.3
                    
                # print(obj_meta.unique_component_id,SECONDARY_DETECTOR_UID_1)
                if(obj_meta.unique_component_id == SECONDARY_DETECTOR_UID_1):
                    bbox_color = "blue"
                    rect_params.has_bg_color = 1
                    rect_params.border_color.red = 0.0
                    rect_params.border_color.green = 0.0
                    rect_params.border_color.blue = 1.0
                    rect_params.border_color.alpha = 1.0
                    rect_params.bg_color.red = 0.0
                    rect_params.bg_color.green = 0.0
                    rect_params.bg_color.blue = 1.0
                    rect_params.bg_color.alpha = 0.3
                    
                if detect_type is not None:
                    text_params.display_text = detect_type
                    
                    if detect_type in anomaly_objs:
                        bbox_color = "red"
                        rect_params.has_bg_color = 1
                        rect_params.border_color.red = 1.0
                        rect_params.border_color.green = 0.0
                        rect_params.border_color.blue = 0.0
                        rect_params.border_color.alpha = 1.0
                        rect_params.bg_color.red = 1.0
                        rect_params.bg_color.green = 0.0
                        rect_params.bg_color.blue = 0.0
                        rect_params.bg_color.alpha = 0.3
                
                parent  = obj_meta.parent

                if parent is not None:
                    obj_id = parent.object_id
                else :
                    obj_id  =  int(obj_meta.object_id)
                
                if(obj_meta.unique_component_id == PRIMARY_DETECTOR_UID_1):
                    if 'Activity' in subscriptions:
                        output_lbl = fetch_activity_info(detect_type)
                    else:
                        output_lbl = fetch_member_info(detect_type)
                        
                    if output_lbl is not None and len(output_lbl)!=0:
                        obj_id_str = str(obj_id)
                        if obj_id_str in output_lbl:
                            text_params.display_text = output_lbl[obj_id_str]
                                
                n_frame_bbox = None
                
                if output_lbl is not None and len(output_lbl)!=0:
                    obj_id_str = str(obj_id)
                    if obj_id_str in output_lbl:
                        n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, output_lbl[obj_id_str], bbox_color)
                    else:
                        n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)
                else:
                    n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)
                    
                        
                cv2.imwrite(f'{infer_path}/{frame_number}.jpg',n_frame_bbox)
                
                n_frame_crop = crop_object(n_frame, obj_meta)
                frame_crop_copy = cv2.cvtColor(n_frame_crop, cv2.COLOR_RGBA2BGRA)
                frame_crop = cv2.cvtColor(frame_crop_copy, cv2.COLOR_BGR2RGB)
                # if(obj_meta.unique_component_id == SECONDARY_DETECTOR_UID_2):
                    # cv2.imwrite(f'{crop_path}/{frame_number}.jpg',frame_crop)


                if obj_id not in age_dict:
                    age_dict[obj_id] = 0
                    age_dict[obj_id] = age_dict[obj_id] + 1
                else:   
                    age_dict[obj_id] = age_dict[obj_id] + 1


                if detect_type in subscriptions_class_list:
                    obj_dict =  {
                    'detect_type' : detect_type,
                    'confidence_score': confidence_score,
                    'obj_id' : obj_id,
                    'bbox_left' : left,
                    'bbox_top' : top,
                    'bbox_right' : left + width,
                    'bbox_bottom' : top + height,
                    'timestamp' :  dt.strftime("%H:%M:%S %d/%m/%Y"),
                    'crop' : cv2.cvtColor(frame_crop_copy, cv2.COLOR_BGR2RGB),
                    'age' : age_dict[obj_id]
                    }
                    if detect_type in constantIdObjects:
                        obj_dict['obj_id'] = 1
                    frame_dict['objects'].append(obj_dict)
                    
            except StopIteration:
                break
            
            obj_counter[obj_meta.class_id] += 1
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
            # detect_data.append(frame_dict)   #optional line detect_data is not used any where just holding all frame_dicts
            if n_frame_bbox is not None:
                # cnt = cnt + 1
                # if cnt >= 25:
                #     cv2.imwrite("test1.jpg",n_frame_bbox)
                frame_dict['np_arr'] = n_frame_bbox   
                frame_dict['org_frame'] = n_frame
            else:
                frame_dict['np_arr'] = frame_copy
                frame_dict['org_frame'] = n_frame
            cnt = cnt + 1

            # print("starting to put elements in queue")
            # print(frame_dict)
            # print(detect_type)
            framedata_queue.put([frame_dict,dev_id_dict])
            
            if is_aarch64():
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps() or decoder_src_pad.query_caps()
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
    uri_decode_bin.set_property("file-loop", True)
    uri_decode_bin.set_property("udp-buffer-size",1048576)
    uri_decode_bin.set_property("num-extra-surfaces",2)
    # uri_decode_bin.set_property("drop-frame-interval",2)
    # uri_decode_bin.set_property("file-loop", "true")
    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        uri_decode_bin.set_property("cudadec-memtype", mem_type)
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

    past_tracking_meta[0]=1
    number_sources=len(args)
    print(number_sources)


    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

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
        uri_name = args[i]['rtsp']

        dev_id_dict[i] = args[i]
        print(dev_id_dict)

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
    
    print("Creating Pgie2 \n ")
    pgie2 = Gst.ElementFactory.make("nvinfer", "primary-inference-2")
    if not pgie2:
        sys.stderr.write(" Unable to create pgie \n")
    

    print("Creating sgie1\n ")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-inference-1")
    if not sgie1:
        sys.stderr.write(" Unable to create sgie1 \n")

    print("Creating sgie2 \n ")
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-inference-2")
    if not sgie2:
        sys.stderr.write(" Unable to create sgie2 \n")    

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

    print("Creating parser \n ")
    parser = Gst.ElementFactory.make("h264parse", "parser") 
    if not parser:
        sys.stderr.write(" Unable to create parser \n")
    
    print("Creating Sink \n")
    sink = Gst.ElementFactory.make("hlssink","sink")
    if not sink:
        sys.stderr.write(" Unable to create sink \n")

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
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=640, height=640")
    capsfilter.set_property("caps", caps)

    queue = Gst.ElementFactory.make("queue", "queue_0")
    queue1 = Gst.ElementFactory.make("queue", "queue_1")
    queue2 = Gst.ElementFactory.make("queue", "queue_2")
    queue3 = Gst.ElementFactory.make("queue", "queue_3")
    queue4 = Gst.ElementFactory.make("queue", "queue_4")
    
    queue.set_property("leaky", 2)
    queue1.set_property("leaky", 2)
    queue2.set_property("leaky", 2)
    queue3.set_property("leaky", 2)
    queue4.set_property("leaky", 2)
    queue.set_property("max-size-buffers", 100)
    queue.set_property("max-size-bytes", 1242880)
    queue.set_property("max-size-time", 100000000)
    # streammux.set_property('config-file-path', 'mux_config.txt')
    # streammux.set_property('batch-size', number_sources)
    streammux.set_property("max-latency", 40000000)
    streammux.set_property('sync-inputs', 1)
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    # streammux.set_property('buffer-pool-size',10)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 1)

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

#############################################################################################################
    pgie1.set_property('config-file-path', pgie1_path)
    pgie1_batch_size=pgie1.get_property("batch-size")
    print("PGIE1 BATCH SIZE: ", pgie1_batch_size)
    # if(pgie1_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",pgie1_batch_size," with number of sources ", number_sources," \n")
    # pgie1.set_property("batch-size", number_sources)

#############################################################################################################
    sgie1.set_property('config-file-path', sgie1_path)
    sgie1_batch_size=sgie1.get_property("batch-size")
    print("SGIE1 BATCH SIZE: ", sgie1_batch_size)
    # if(sgie1_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",sgie1_batch_size," with number of sources ", number_sources," \n")
    # sgie1.set_property("batch-size", number_sources)

#############################################################################################################    
    sgie2.set_property('config-file-path', sgie2_path)
    sgie2_batch_size=sgie2.get_property("batch-size")
    print("SGIE2 BATCH SIZE: ", sgie2_batch_size)
    # if(sgie2_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",sgie2_batch_size," with number of sources ", number_sources," \n")
    # sgie2.set_property("batch-size", number_sources)

#############################################################################################################    
    pgie2.set_property('config-file-path', pgie2_path)
    pgie2_batch_size=pgie2.get_property("batch-size")
    print("PGIE2 BATCH SIZE: ", pgie2_batch_size)
    # if(pgie2_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",pgie2_batch_size," with number of sources ", number_sources," \n")
    # pgie2.set_property("batch-size", number_sources)

    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        
    #adding elements to the pipeline
    print("Adding elements to Pipeline \n")
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
    
    #linking all the elements
    print("Linking elements in the Pipeline \n")
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
    

    # try:
    #     for i in range(number_sources):
    #         try:
    #             logger.info("Creating sink %d\n",i)
                
    #             if args[i]['ddns'] is None or args[i]['ddns'] == "":
    #                 DDNS = ddns_name
    #             else:
    #                 DDNS = args[i]['ddns']
                    
    #             video_info = hls_path + '/' + dev_id_dict[i]['deviceId']
    #             if not os.path.exists(video_info):
    #                 os.makedirs(video_info, exist_ok=True)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink parameters", exc_info=e)
                
    #         try:
    #             logger.info("Creating hlssink %d\n",i)
    #             sink = Gst.ElementFactory.make("hlssink", f"sink_{i}")
    #             if not sink:
    #                 sys.stderr.write(" Unable to create hlssink \n")
    #             pipeline.add(sink)
    #             devid = dev_id_dict[i]['deviceId']
    #             sink.set_property('playlist-root', f'https://{DDNS}/live/{devid}') # Location of the playlist to write
    #             # sink.set_property('playlist-root', f'http://localhost:9001/{devid}') # Location of the playlist to write
                
    #             sink.set_property('playlist-location', f'{video_info}/{devid}.m3u8') # Location where .m3u8 playlist file will be stored
    #             sink.set_property('location',  f'{video_info}/segment.%01d.ts')  # Location whee .ts segmentrs will be stored
    #             sink.set_property('target-duration', target_duration) # The target duration in seconds of a segment/file. (0 - disabled, useful
    #             sink.set_property('playlist-length', playlist_length) # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
    #             sink.set_property('max-files', max_files) # Maximum number of files to keep on disk. Once the maximum is reached,old files start to be deleted to make room for new ones.
    #         except Exception as e:
    #             logger.error("An error occurred while creating and setting properties of hlssink", exc_info=e)
            
    #         try:
    #             # creating queue
    #             queue_0 = Gst.ElementFactory.make("queue", f"queue_1_{i}")
    #             pipeline.add(queue_0)
    #             queue_1 = Gst.ElementFactory.make("queue", f"queue_2_{i}")
    #             pipeline.add(queue_1)
    #             queue_2 = Gst.ElementFactory.make("queue", f"queue_3_{i}")
    #             pipeline.add(queue_2)
    #             queue_3 = Gst.ElementFactory.make("queue", f"queue_4_{i}")
    #             pipeline.add(queue_3)
    #             queue_4 = Gst.ElementFactory.make("queue", f"queue_5_{i}")
    #             pipeline.add(queue_4)
    #             queue_5 = Gst.ElementFactory.make("queue", f"queue_6_{i}")
    #             pipeline.add(queue_5)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink queues", exc_info=e)
                
    #         try:
    #             logger.info("Creating Sink h264parse %d\n",i)
    #             parser = Gst.ElementFactory.make("h264parse", f"parse_{i}")
    #             if not parser:
    #                 sys.stderr.write(" Unable to create h264parse \n")
    #             pipeline.add(parser)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink h264parse", exc_info=e)
                
    #         try:
    #             # creating nvvidconv
    #             logger.info("Creating Sink nvvidconv %d\n",i)
    #             nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvc_{i}")
    #             if not nvvideoconvert:
    #                 sys.stderr.write(" Unable to create nvvidconv \n")
    #             pipeline.add(nvvideoconvert)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink nvvidconv", exc_info=e)
                
    #         try:
    #             logger.info("Creating Sink OSD capsfilter %d\n",i)
    #             capsfilter_osd = Gst.ElementFactory.make("capsfilter", f"caps_osd_{i}")
    #             if not capsfilter_osd:
    #                 sys.stderr.write(" Unable to create sink capsfilter_osd \n")
    #             pipeline.add(capsfilter_osd)
    #             caps_osd = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1280, height=720")
    #             capsfilter_osd.set_property("caps", caps_osd)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink capsfilter_osd", exc_info=e)
            
    #         try:
    #             # creating nvosd
    #             logger.info("Creating Sink OSD %d\n",i)
    #             nvdsosd = Gst.ElementFactory.make("nvdsosd", f"osd_{i}")
    #             if not nvdsosd:
    #                 sys.stderr.write(" Unable to create nvdsosd \n")
    #             pipeline.add(nvdsosd)
    #             nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
    #             nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink nvdsosd", exc_info=e)
            
    #         try:
    #             logger.info("Creating Sink nvvidconv_postosd %d\n",i)
    #             nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", f"nv_post_ods{i}")
    #             if not nvvidconv_postosd:
    #                 sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    #             pipeline.add(nvvidconv_postosd)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink nvvidconv_postosd", exc_info=e)
            
    #         try:
    #             logger.info("Creating Sink capsfilter %d\n",i)
    #             capsfilter = Gst.ElementFactory.make("capsfilter", f"caps_{i}")
    #             if not capsfilter:
    #                 sys.stderr.write(" Unable to create capsfilter \n")
    #             pipeline.add(capsfilter)
    #             caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720")
    #             capsfilter.set_property("caps", caps)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink capsfilter", exc_info=e)
            
    #         try:
    #             logger.info("Creating Sink encoder %d\n",i)
    #             encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{i}") # nvv4l2h264enc
    #             if not encoder:
    #                 sys.stderr.write(" Unable to create Sink nvv4l2h264enc \n")
    #             pipeline.add(encoder)
    #             encoder.set_property("bitrate", hls_bitrate)
    #             encoder.set_property("preset-id", preset_level)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink encoder", exc_info=e)
            
    #         try:
    #             logger.info("Creating Sink mux %d\n",i)
    #             container = Gst.ElementFactory.make("mpegtsmux", f"mux_{i}")
    #             if not container:
    #                 sys.stderr.write(" Unable to create Sink mpegtsmux \n")
    #             pipeline.add(container)
    #         except Exception as e:
    #             logger.error("An error occurred while creating sink mux", exc_info=e)
            
    #         try:
    #             if not is_aarch64():
    #                 mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    #                 nvvideoconvert.set_property("nvbuf-memory-type", mem_type)
    #                 nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)
    #         except Exception as e:
    #             logger.error("An error occurred while creating mem_type for sink", exc_info=e)
            
    #         try:
    #             # connect nvstreamdemux -> queue
    #             padname = "src_%u" %i
    #             demuxsrcpad = demux.get_request_pad(padname)
    #             if not demuxsrcpad:
    #                 sys.stderr.write("Unable to create demux src pad \n")
    #             queuesinkpad = queue_0.get_static_pad("sink")
    #             if not queuesinkpad:
    #                 sys.stderr.write("Unable to create queue sink pad \n")
    #             demuxsrcpad.link(queuesinkpad)
    #         except Exception as e:
    #             logger.error("An error occurred while connecting demux to queue", exc_info=e)
                
    #         try:
    #             queue_0.link(nvvideoconvert)
    #             nvvideoconvert.link(capsfilter_osd)
    #             capsfilter_osd.link(nvdsosd)
    #             nvdsosd.link(queue_1)
    #             queue_1.link(nvvidconv_postosd)
    #             nvvidconv_postosd.link(queue_2)
    #             queue_2.link(capsfilter)
    #             capsfilter.link(queue_3)
    #             queue_3.link(encoder)
    #             encoder.link(parser)
    #             parser.link(queue_4)
    #             queue_4.link(container)
    #             container.link(queue_5)
    #             queue_5.link(sink)
    #         except Exception as e:
    #             logger.error("An error occurred while connecting sink elements in the Pipeline", exc_info=e)
            
    # except Exception as e:
    #     logger.error("An error occurred while creating sink pipeline", exc_info=e)



    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    
    tracker_src_pad=pgie2.get_static_pad("src")
    if not tracker_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, args)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
        
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
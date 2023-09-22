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
import datetime
import pyds
import pytz
# import datetime
import numpy as np
import cv2
from dotenv import load_dotenv
import lmdb
import json
import queue
import threading

from modules.gif.gif_creation import gif_build
from modules.db.db_push import gif_push, gst_hls_push
from modules.components.generate_crop import save_one_box

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
class_bbox_color = ast.literal_eval(os.getenv("class_color"))

PRIMARY_DETECTOR_UID_1 = int(os.getenv("PRIMARY_DETECTOR_UID_1"))
PRIMARY_DETECTOR_UID_2 = int(os.getenv("PRIMARY_DETECTOR_UID_2"))
SECONDARY_DETECTOR_UID_1 = int(os.getenv("SECONDARY_DETECTOR_UID_1"))
SECONDARY_DETECTOR_UID_2 = int(os.getenv("SECONDARY_DETECTOR_UID_2"))
MAX_NUM_SOURCES = int(os.getenv("MAX_NUM_SOURCES"))
OSD_PROCESS_MODE = int(os.getenv("OSD_PROCESS_MODE"))
OSD_DISPLAY_TEXT = int(os.getenv("OSD_DISPLAY_TEXT"))

rtsp_reconnect_interval = int(os.getenv("rtsp_reconnect_interval"))
file_loop = bool(os.getenv("file_loop"))
latency = int(os.getenv("latency"))
num_extra_surfaces = int(os.getenv("num_extra_surfaces"))
udp_buffer_size = int(os.getenv("udp_buffer_size"))
drop_frame_interval = int(os.getenv("drop_frame_interval"))
select_rtp_protocol = int(os.getenv("select_rtp_protocol"))
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
pgie1_path = cwd + f"/models_deepstream/{tenant_name}/{device}/gender/config.txt"
pgie2_path = cwd + f"/models_deepstream/{tenant_name}/{device}/fire/config.txt"
sgie1_path = cwd + f"/models_deepstream/{tenant_name}/{device}/face_detect/config.txt"
sgie2_path = cwd + f"/models_deepstream/{tenant_name}/{device}/dangerous_object/config.txt"
frame_path = cwd + "/static/frames/"
infer_path = cwd + "/static/image/"
crop_path = cwd + "/static/crops/"

framedata_queue = queue.Queue()
age_dict = {}
dev_id_dict = {}
dev_status = {}
gif_dict = {}
gif_created = {}

# past_tracking_meta=[0]

streammux = None
pipeline = None

db_env = lmdb.open(lmdb_path+'/face-detection',
                max_dbs=10)
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
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    
    obj_name = obj_meta.obj_label
    x1 = left
    y1 = top
    x2 = left + width
    y2 = top + height
    crop_img=save_one_box([x1,y1,x2,y2],image)
    crop=cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
    # crop_img = image[top:top+height, left:left+width]
    return crop

def findClassList(subscriptions):
    subscriptions_class_list = [item for sublist in [classDict[each] for each in subscriptions if each in classDict] for item in sublist]
    for each in obj_det_labels:
        subscriptions_class_list.append(each)
    return subscriptions_class_list
    
def fetch_activity_info(detect_type):
    key = "IdLabelInfo"
    output_dict = {}
    with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
        value = db_txn.get(key.encode())
        if value is not None:
            data = json.loads(value.decode())
            for key, value in data.items():
                memID = value['memberID']
                activities = list(set(value['activity']))
                
                if memID is not None:
                    if memID == '100':
                        if memID in track_type:
                            member_track_type = track_type[memID]
                    else:
                        mem_type = memID[:2]
                        if mem_type in track_type:
                            member_track_type = track_type[mem_type]
                else:
                    member_track_type = None
                
                if member_track_type is None:
                    sentence = f"{detect_type} {' '.join(activities)}"
                else:
                    sentence = f"{member_track_type} {detect_type} {' '.join(activities)}"

                output_dict[key] = sentence
            
            return output_dict
        else:
            return None
        
def fetch_member_info(detect_type):
    key = "trackIdMemIdDict"
    output_dict = {}
    with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
        value = db_txn.get(key.encode())
        if value is not None:
            data = json.loads(value.decode())
            for key, value in data.items():
                memID = (data[key])[-1]
                
                if memID is not None:
                    if memID == '100':
                        if memID in track_type:
                            member_track_type = track_type[memID]
                    else:
                        mem_type = memID[:2]
                        if mem_type in track_type:
                            member_track_type = track_type[mem_type]
                else:
                    member_track_type = None
                
                if member_track_type is None:
                    sentence = f"{detect_type}"
                else:
                    sentence = f"{member_track_type} {detect_type}"

                output_dict[key] = sentence
            return output_dict
        else:
            return None

def tracker_src_pad_buffer_probe(pad,info,dev_list):
    
    global gif_dict
    # frame_number=0

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
        l_obj=frame_meta.obj_meta_list
        n_frame_bbox = None
        output_lbl = None
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data) #new obj
                confidence_score = obj_meta.confidence
                detect_type = obj_meta.obj_label
                rect_params = obj_meta.rect_params
                text_params = obj_meta.text_params
                left = rect_params.left
                top = rect_params.top
                width = rect_params.width
                height = rect_params.height
                bbox_color = None
                text_params.display_text = detect_type
                
                for key, value in class_bbox_color.items():
                    if detect_type in value:
                        bbox_color = key
                        break
                    
                if bbox_color:
                    rect_params.has_bg_color = 1
                    rect_params.border_color.red = 0.0
                    rect_params.border_color.green = 0.0
                    rect_params.border_color.blue = 0.0
                    rect_params.bg_color.red = 0.0
                    rect_params.bg_color.green = 0.0
                    rect_params.bg_color.blue = 0.0
                    rect_params.border_color.alpha = 1.0
                    rect_params.bg_color.alpha = 0.3
                    if bbox_color == "green":
                        rect_params.border_color.green = 1.0
                        rect_params.bg_color.green = 1.0
                    if bbox_color == "blue":
                        rect_params.border_color.blue = 1.0
                        rect_params.bg_color.blue = 1.0
                    if bbox_color == "red":
                        rect_params.border_color.red = 1.0
                        rect_params.bg_color.red = 1.0
                
                parent  = obj_meta.parent

                if parent is not None:
                    obj_id = parent.object_id
                else :
                    obj_id = int(obj_meta.object_id)
                
                if(obj_meta.unique_component_id == PRIMARY_DETECTOR_UID_1):
                    if 'Activity' in subscriptions:
                        output_lbl = fetch_activity_info(detect_type)
                    else:
                        output_lbl = fetch_member_info(detect_type)
                        
                    if output_lbl is not None:
                        if len(output_lbl)!=0:
                            obj_id_str = str(obj_id)
                            if obj_id_str in output_lbl:
                                text_params.display_text = output_lbl[obj_id_str]
                                
                # n_frame_bbox = None
                
                if output_lbl is not None and len(output_lbl)!=0:
                    obj_id_str = str(obj_id)
                    if obj_id_str in output_lbl:
                        n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, output_lbl[obj_id_str], bbox_color)
                    else:
                        n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)
                else:
                    n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)  
                # cv2.imwrite(f'{infer_path}/{frame_number}.jpg',n_frame_bbox)
                
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
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
            if n_frame_bbox is not None:
                frame_dict['np_arr'] = n_frame_bbox   
                frame_dict['org_frame'] = n_frame
            else:
                frame_dict['np_arr'] = frame_copy
                frame_dict['org_frame'] = n_frame

            print("starting to put elements in queue")
            # print(frame_dict)
            # print(detect_type)
            framedata_queue.put([frame_dict,dev_id_dict])
            
            if is_aarch64():
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
 
def stop_release_source(source_id):
    
    global streammux
    global pipeline
    global dev_status

    #Attempt to change status of source to be released 
    state_return = dev_status[source_id][1].set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")
        pad_name = "sink_%u" % source_id
        print(pad_name)
        #Retrieve sink pad to be released
        sinkpad = streammux.get_static_pad(pad_name)
        #Send flush stop event to the sink pad, then release from the streammux
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE SUCCESS\n")
        #Remove the source bin from the pipeline
        pipeline.remove(dev_status[source_id][1])

    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")
    
    elif state_return == Gst.StateChangeReturn.ASYNC:
        state_return = dev_status[source_id][1].get_state(Gst.CLOCK_TIME_NONE)
        pad_name = "sink_%u" % source_id
        print(pad_name)
        sinkpad = streammux.get_static_pad(pad_name)
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE ASYNC\n")
        pipeline.remove(dev_status[source_id][1])
 
def delete_sources(delete_dev):
    
    global dev_id_dict
    deviceId = delete_dev['deviceId']
    source_id = None
    for key, value in dev_id_dict.items():
        if value['deviceId'] == deviceId:
            source_id = key
            break

    #Choose an enabled source to delete
    #Disable the source
    print("SOURCE ENABLED: ", dev_status[source_id][0])
    dev_status[source_id][0] = False
    #Release the source
    print("Calling Stop %d " % source_id)
    stop_release_source(source_id)
    if(dev_status[source_id][0]) == False:
        dev_status[source_id][1] = None
        
    # #Quit if no sources remaining
    # if (g_num_sources == 0):
    #     loop.quit()
    #     print("All sources stopped quitting")
    #     return False

    return True

def add_sources(add_dev):
    global dev_status
    source_id = None
    if((len(dev_status) != 0) and (len(dev_status) < MAX_NUM_SOURCES)):
        for key, value in dev_status.items():
            if not value[0]:
                source_id = key
                print(dev_status)
                break
        if source_id == None:
            source_id = len(dev_status)
            dev_status[source_id] = [None, None]
            print(dev_status)
    
        print("Adding a new device %d " % source_id)
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
        print(dev_status)
        dev_status[source_id][0] = True
        dev_status[source_id][1] = source_bin
        print(dev_status)
    else:
        print("DEVICE LIST IS EITHER EMPTY OR EXCEEDED THE LIMIT!!")
        print("DEVICE ADDITION FAILED")

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

    # if "source" in name:
    #     source_element = child_proxy.get_by_name("source")
    #     if source_element.find_property('drop-on-latency') != None:
    #         Object.set_property("drop-on-latency", True)
            
    # if(name.find("nvv4l2decoder") != -1):
    #     if (is_aarch64()):
    #         Object.set_property("enable-max-performance", True)
    #         Object.set_property("drop-frame-interval", 0)
    #         Object.set_property("num-extra-surfaces", 0)

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
    uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{index}")
    uri_decode_bin.set_property("rtsp-reconnect-interval", rtsp_reconnect_interval)
    uri_decode_bin.set_property("file-loop", file_loop)
    # uri_decode_bin.set_property("udp-buffer-size",1048576)
    uri_decode_bin.set_property("latency", latency)
    uri_decode_bin.set_property("num-extra-surfaces", num_extra_surfaces)
    uri_decode_bin.set_property("udp-buffer-size", udp_buffer_size)
    uri_decode_bin.set_property("drop-frame-interval", drop_frame_interval)
    uri_decode_bin.set_property("select-rtp-protocol", select_rtp_protocol)
    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        uri_decode_bin.set_property("cudadec-memtype", mem_type)
    # else:
    #     uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    # if not uri_decode_bin:
    #     sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
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

def main(server, args):
    
    print(args)
    global dev_id_dict
    global dev_status
    global pipeline
    global streammux
    
    # Check input arguments
    # past_tracking_meta[0]=1
    number_sources=len(args)
    print("NUMBER OF SOURCES: ", number_sources)
    # Standard GStreamer initialization
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
        if(len(dev_status)) < MAX_NUM_SOURCES:
            print("Creating source_bin ",i," \n ")
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
            print("DEVICE LIMIT IS EXCEEDED!! CANNOT ADD ANYMORE DEVICES")

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
    
    queue.set_property("leaky", leaky)
    queue1.set_property("leaky", leaky)
    queue2.set_property("leaky", leaky)
    queue3.set_property("leaky", leaky)
    queue4.set_property("leaky", leaky)
    queue.set_property("max-size-buffers", max_size_buffers)
    queue.set_property("max-size-bytes", max_size_bytes)
    queue.set_property("max-size-time", max_size_time)
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

    pgie1.set_property('config-file-path', pgie1_path)
    pgie1_batch_size=pgie1.get_property("batch-size")
    print("PGIE1 BATCH SIZE: ", pgie1_batch_size)
    # if(pgie1_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",pgie1_batch_size," with number of sources ", number_sources," \n")
    # pgie1.set_property("batch-size", number_sources)

    sgie1.set_property('config-file-path', sgie1_path)
    sgie1_batch_size=sgie1.get_property("batch-size")
    print("SGIE1 BATCH SIZE: ", sgie1_batch_size)
    # if(sgie1_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",sgie1_batch_size," with number of sources ", number_sources," \n")
    # sgie1.set_property("batch-size", number_sources)

    sgie2.set_property('config-file-path', sgie2_path)
    sgie2_batch_size=sgie2.get_property("batch-size")
    print("SGIE2 BATCH SIZE: ", sgie2_batch_size)
    # if(sgie2_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",sgie2_batch_size," with number of sources ", number_sources," \n")
    # sgie2.set_property("batch-size", number_sources)

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
    
    for i in range(number_sources):
        print("Creating sink ",i," \n ")

        DDNS = args[i]['ddns']
        if DDNS is None or " ":
            DDNS = ddns_name

        video_info = hls_path + '/' + dev_id_dict[i]['deviceId']
        if not os.path.exists(video_info):
            os.makedirs(video_info, exist_ok=True)
        
        sink = Gst.ElementFactory.make("hlssink", f"sink_{i}")
        pipeline.add(sink)
        devid = dev_id_dict[i]['deviceId']
        sink.set_property('playlist-root', f'https://{DDNS}/live/{devid}') # Location of the playlist to write
        # sink.set_property('playlist-root', f'http://localhost:9001/{devid}') # Location of the playlist to write
        
        sink.set_property('playlist-location', f'{video_info}/{devid}.m3u8') # Location where .m3u8 playlist file will be stored
        sink.set_property('location',  f'{video_info}/segment.%01d.ts')  # Location whee .ts segmentrs will be stored
        sink.set_property('target-duration', target_duration) # The target duration in seconds of a segment/file. (0 - disabled, useful
        sink.set_property('playlist-length', playlist_length) # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
        sink.set_property('max-files', max_files) # Maximum number of files to keep on disk. Once the maximum is reached,old files start to be deleted to make room for new ones.
        
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
        
        capsfilter_osd = Gst.ElementFactory.make("capsfilter", f"caps_osd_{i}")
        pipeline.add(capsfilter_osd)
        caps_osd = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=1280, height=720")
        capsfilter_osd.set_property("caps", caps_osd)
        
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
        encoder.set_property("bitrate", hls_bitrate)
        # if is_aarch64():
        #     encoder.set_property("preset-level", "FastPreset")
        # else:
        encoder.set_property("preset-id", preset_level)
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
        nvvideoconvert.link(capsfilter_osd)
        capsfilter_osd.link(nvdsosd)
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
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # # # GLib.timeout_add_seconds(200, delete_sources, 5)
    # dev_data = {'deviceId': 'c7551495-d8b7-4f71-bfc7-5581a9da9cb2', 'tenantId': 'a8aa8168-df3b-4f01-8836-44407e8b14d1', 'urn': 'uuid:626d6410-d723-4dc8-o867-e943c0987dcb', 'ddns': None, 'ip': '0.0.0.1', 'port': 1234, 'videoEncodingInformation': 'H265', 'username': 'happymonk', 'rtsp': 'rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif', 'password': 'admin123', 'subscriptions': ['Activity', 'Fire/Smoke', 'Dangerous-Object'], 'lat': 26.25, 'long': 88.11}
    # delete_sources(dev_data)
    # dev_data_1 = {'deviceId': '8ed7d76c-7863-41fe-8c44-b6633760bc4e', 'tenantId': 'a8aa8168-df3b-4f01-8836-44407e8b14d1', 'urn': 'uuid:626d6410-d723-4dc8-o867-e943c0987dcb', 'ddns': None, 'ip': '0.0.0.1', 'port': 1234, 'videoEncodingInformation': 'H265', 'username': 'happymonk', 'rtsp': 'rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif', 'password': 'admin123', 'subscriptions': ['Activity', 'Fire/Smoke', 'Dangerous-Object'], 'lat': 26.25, 'long': 88.11}
    # add_sources(dev_data_1)
    # dev_data_2 = {'deviceId': '4ec7d76c-7863-41fe-8c44-b6633760bc4e', 'tenantId': 'a8aa8168-df3b-4f01-8836-44407e8b14d1', 'urn': 'uuid:626d6410-d723-4dc8-o867-e943c0987dcb', 'ddns': None, 'ip': '0.0.0.1', 'port': 1234, 'videoEncodingInformation': 'H265', 'username': 'happymonk', 'rtsp': 'rtsp://admin:admin123@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif', 'password': 'admin123', 'subscriptions': ['Activity', 'Fire/Smoke', 'Dangerous-Object'], 'lat': 26.25, 'long': 88.11}
    # add_sources(dev_data_2)
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
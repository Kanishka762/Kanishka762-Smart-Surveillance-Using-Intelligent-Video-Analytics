from modules.components.load_paths import *
from init import loadLogger
from dotenv import load_dotenv
logger = loadLogger()
load_dotenv(dotenv_path)
import cv2
import ast
import datetime
import pytz

from modules.deepstream.genCropsBBox import draw_bounding_boxes,crop_object
from modules.lmdbSubmodules.liveStreamLabelGen import classLabels

class_bbox_color = ast.literal_eval(os.getenv("class_color"))
constantIdObjects = ast.literal_eval(os.getenv("constantIdObjects"))
place = os.getenv("place")
timezone = pytz.timezone(f'{place}')  #assign timezone
PRIMARY_DETECTOR_UID_1 = int(os.getenv("PRIMARY_DETECTOR_UID_1"))
PRIMARY_DETECTOR_UID_2 = int(os.getenv("PRIMARY_DETECTOR_UID_2"))
SECONDARY_DETECTOR_UID_1 = int(os.getenv("SECONDARY_DETECTOR_UID_1"))
SECONDARY_DETECTOR_UID_2 = int(os.getenv("SECONDARY_DETECTOR_UID_2"))

crop_path = f"{cwd}/static/crops/"

dt = datetime.datetime.now(timezone)
age_dict = {}

def frame_color_convert(frame, color_format):
    try:
        if frame is not None and frame.size != 0:
            if color_format == "BGR2RGB":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if color_format == "RGBA2RGB":
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            if color_format == "RGBA2BGRA":
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            return frame
    except Exception as e:
        logger.error("An error occurred in color conversion", exc_info=e)
    
def fetch_subsriptions(device_dict, camera_id):
    try:
        dev_info = device_dict[camera_id]
        return dev_info['subscriptions']
    except Exception as e:
        logger.error("An error occurred while fetching Subscriptions", exc_info=e)
        
def initial_frame_dict(frame_number, num_detect, camera_id, device_timestamp):
    try:
        return {
            'frame_number': frame_number,
            'total_detect': num_detect,
            'camera_id': camera_id,
            'frame_timestamp': device_timestamp,
            'objects': [],  # List to hold object dictionaries
        }
    except Exception as e:
        logger.error("An error occurred while initializing frame dict", exc_info=e)
    
def fetch_bbox_color(detect_type):
    try:
        for key, value in class_bbox_color.items():
            if detect_type in value:
                bbox_color = key
                break
        return bbox_color
    except Exception as e:
        logger.error("An error occurred while fetching bbox color", exc_info=e)

def update_bbox_color(bbox_color, rect_params):
    try:
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
            if bbox_color == "blue":
                rect_params.border_color.blue = 1.0
                rect_params.bg_color.blue = 1.0
            elif bbox_color == "green":
                rect_params.border_color.green = 1.0
                rect_params.bg_color.green = 1.0
            elif bbox_color == "red":
                rect_params.border_color.red = 1.0
                rect_params.bg_color.red = 1.0
    except Exception as e:
        logger.error("An error occurred while assigning bbox color in rect_params", exc_info=e)

def update_frame_label(obj_meta, subscriptions, detect_type, text_params, obj_id, frame_copy, bbox_color, frame_number):
    # sourcery skip: lift-return-into-if
    n_frame_bbox = None
    output_lbl = None
    classLabelsobj = classLabels()
    try:
        if(obj_meta.unique_component_id == PRIMARY_DETECTOR_UID_1):
            if 'Facial-Recognition' in subscriptions and len(subscriptions) == 1:
                output_lbl = classLabelsobj.fetch_member_info(detect_type)
            elif 'Activity' in subscriptions:
                output_lbl = classLabelsobj.fetch_activity_info(detect_type)

            if output_lbl is not None and len(output_lbl)!=0 and str(obj_id) in output_lbl:
                text_params.display_text = output_lbl[str(obj_id)]
    except Exception as e:
        logger.error("An error occurred while fetching frame label", exc_info=e)

    try:
        if output_lbl is None or len(output_lbl) == 0:
            n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)
        elif str(obj_id) in output_lbl:
            n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, output_lbl[str(obj_id)], bbox_color)
        else:
            n_frame_bbox = draw_bounding_boxes(frame_copy, obj_meta, obj_meta.confidence, detect_type, bbox_color)
        return n_frame_bbox
    except Exception as e:
        logger.error("An error occurred while drawing labels on frames", exc_info=e)
        
def get_object_info(obj_meta):
    try:
        confidence_score = obj_meta.confidence
        detect_type = obj_meta.obj_label
        rect_params = obj_meta.rect_params
        text_params = obj_meta.text_params
        parent  = obj_meta.parent
        left = rect_params.left
        top = rect_params.top
        width = rect_params.width
        height = rect_params.height
        bbox_color = None
        text_params.display_text = detect_type
        return confidence_score, detect_type, rect_params, text_params, parent, left, top, width, height, bbox_color
    except Exception as e:
        logger.error("An error occurred while getting Object Meta information", exc_info=e)
    
def create_obj_dict(n_frame, obj_meta, obj_id, detect_type, confidence_score, left, top, width, height, frame_dict, subscriptions_class_list, frame_number):
    try:
        n_frame_crop = crop_object(n_frame, obj_meta)
        frame_crop_copy = frame_color_convert(n_frame_crop, "RGBA2BGRA")
        # cv2.imwrite(f'{crop_path}/{frame_number}.jpg',frame_crop_copy)
    except Exception as e:
        logger.error("An error occurred while cropping frames", exc_info=e)

    try:
        if obj_id not in age_dict:
            age_dict[obj_id] = 0
        age_dict[obj_id] = age_dict[obj_id] + 1
    except Exception as e:
        logger.error("An error occurred while calculating age factor", exc_info=e)
    
    try:
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
            'crop' : frame_color_convert(frame_crop_copy, "BGR2RGB"),
            'age' : age_dict[obj_id]
            }
            if detect_type in constantIdObjects:
                obj_dict['obj_id'] = 1
            if frame_dict is not None:
                frame_dict['objects'].append(obj_dict)
        return frame_dict
    except Exception as e:
        logger.error("An error occurred while creating object dictionary", exc_info=e)
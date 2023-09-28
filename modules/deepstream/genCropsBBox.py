from modules.components.load_paths import *

import ast
import os
import cv2
from dotenv import load_dotenv
load_dotenv(dotenv_path)

from modules.components.generate_crop import save_one_box


frame_bbox_color = ast.literal_eval(os.getenv("color"))


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
    return cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)

from io import BytesIO
import face_recognition 
import subprocess as sp
import cv2
from datetime import datetime  #datetime module to fetch current time when frame is detected
import numpy as np
from pytz import timezone 
from nanoid import generate
import random
import string
# from main import gen_datainfo
import uuid
# from retinaface.pre_trained_models import get_model

# model_face = get_model("resnet50_2020-07-20", max_size=2048)
# model_face.eval()
from modules.alarm.alarm_light_trigger import alarm
face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2


# def face_rec(image):
#     face_coords = []
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     annotation = model_face.predict_jsons(image)
#     print(annotation)
#     if len(annotation[0]['bbox'])>0:
#         for item in annotation:
#             box = item['bbox']
#             print(box)
#             x1,y1,x2,y2 = box
#             crop = image[int(y1):int(y2),int(x1):int(x2)]
#             # characters = string.ascii_letters + string.digits
#             # random_string = ''.join(random.choices(characters, k=5))
#             # cv2.imwrite("/home/agx123/DS_pipeline_new/face_crops/"+random_string+".jpg",crop)


#             bbox = []
#             for i in box:
#                 i = int(i)
#                 bbox.append(i)
#             if len(bbox) > 0: 
#                 first_element = bbox.pop(0)
#                 bbox.append(first_element)
#                 face_coords.append(tuple(bbox))
        
#         return face_coords
#     else:
#         return []


def find_person_type(im0,datainfo):
    # characters = string.ascii_letters + string.digits
    # random_string = ''.join(random.choices(characters, k=5))
    # cv2.imwrite("/home/agx123/DS_pipeline_new/body_crops/"+random_string+".jpg",im0)

    known_whitelist_faces = datainfo[0]
    known_blacklist_faces = datainfo[1]
    unknown_faces = []
    unknown_id = []
    known_whitelist_id = datainfo[2]
    known_blacklist_id = datainfo[3]
    minimum_distance = []
    np_arg_src_list = known_whitelist_faces + known_blacklist_faces
    np_bytes2 = BytesIO()
    # np.save(np_bytes2, im0, allow_pickle=True)
    # np_bytes2 = np_bytes2.getvalue()
    # image = cv2.imread(im0) # if im0 does not work, try with im1
    image = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(image)
    # locations = face_rec(image)
    # print(locations)
    encodings = face_recognition.face_encodings(image,locations)

    did = None
    track_type = "100"

    if len(locations) != 0:
        print(len(known_whitelist_faces))
        if len(known_whitelist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_white = face_recognition.compare_faces(known_whitelist_faces,face_encoding)
                faceids_white = face_recognition.face_distance(known_whitelist_faces,face_encoding)
                print(faceids_white)

                matchindex_white = np.argmin(faceids_white)
                if min(faceids_white) <=0.60:
                    print(min(faceids_white), matches_white[matchindex_white])

                    if matches_white[matchindex_white]:
                        
                    
                        did = '00'+ str(known_whitelist_id[matchindex_white])
                        track_type = "00"
                        return(did , track_type)

        print(len(known_blacklist_faces))
        if len(known_blacklist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_black = face_recognition.compare_faces(known_blacklist_faces,face_encoding)
                faceids_black = face_recognition.face_distance(known_blacklist_faces,face_encoding)
                print(faceids_black)
                matchindex_black = np.argmin(faceids_black)
 
                if min(faceids_black) <=0.60:
                    print(min(faceids_black), matches_black[matchindex_black])
                    # print()

                    if matches_black[matchindex_black]:
                        alarm()
                        did = '01'+ str(known_blacklist_id[matchindex_black])
                        track_type = "01"
                        return(did , track_type)

        print(len(unknown_faces))
        if len(unknown_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                matches_unknown = face_recognition.compare_faces(unknown_faces,face_encoding)
                faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                matchindex_unknown = np.argmin(faceids_unknown)
                minimum_distance.append(min(faceids_unknown))
                if min(faceids_unknown) <=0.60:
                    print(min(faceids_unknown), matches_unknown[matchindex_unknown])

                    if matches_unknown[matchindex_unknown]:
                    
                        did = "11" + str(unknown_id[matchindex_unknown])
                        track_type = "11"
                        return(did , track_type)


        else:
            unknown_faces.append(encodings)
            id = str(uuid.uuid4())
            did = id
            track_type = "10"

            if id not in unknown_id:
                unknown_id.append(id)

            return(did , track_type)
            



    return did, track_type

# im0 = cv2.imread("/home/nivetheni/TCI_express_srihari/TCI_express/image/QmWzcNaQTmrUsaswdaTibkyuc3HYTCSfsr9wTtjNtfhfEq.jpg")
# print(find_person_type(im0,datainfo))

    
    
    
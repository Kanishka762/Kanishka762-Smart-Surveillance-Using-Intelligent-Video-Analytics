from modules.components.load_paths import *
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
from pathlib import Path
import glob
import os
import subprocess as sp
import os
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")

env = lmdb.open(lmdb_path+'/face-detection.lmdb',
                max_dbs=10, map_size=int(100e9))

known_db = env.open_db(b'white_list')
unknown_db = env.open_db(b'black_list')

def conv_img2bytes(image_path):
    image  = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    encodedNumpyData = cv2.imencode('.jpg', image)[1].tobytes()
    return encodedNumpyData

def insert_db(memberId, person_img_bytes, db_):
    with env.begin(write=True) as txn:
        txn.put(bytearray(memberId, "utf-8"), person_img_bytes, db=db_)
    return True

def cid_to_image(cid):
    global image_path
    #'ipfs --api={ipfs_url} add {file_path} -Q'.format(ipfs_url=ipfs_url, file_path=src_file)
    command = 'ipfs --api={ipfs_url} get {cid}'.format(ipfs_url=ipfs_url,cid=cid)
    output = sp.getoutput(command)
    im_path = image_path+'/'+str(cid)+".jpg"
    os.rename(cid, im_path)
    return im_path

def add_member_to_lmdb(MemberPublish):
    list_of_members =  MemberPublish["member"]
    for each_member in list_of_members:
        print(each_member)
        faceCID = each_member["faceCID"]
        print(faceCID)
        memberId = each_member["memberId"]
        print(memberId)
        class_type = each_member["type"]
        print(class_type)
        image_path = cid_to_image(faceCID[0])
        print(image_path)
        person_img_bytes = conv_img2bytes(image_path)
        if class_type == "known":
            db_ = known_db
            print("known_db")
        else:
            db_ = unknown_db
            print("unknown_db")

        status  = insert_db(memberId, person_img_bytes, db_)
        print("inserted",status)
        print("_______________________________________")
        return status

# # #get json from nats and call add_member_to_lmdb(nats_json)
# MemberPublish = {'id': 'ui75LlKf6gzrfa7LuU2y27Jaq1nxO2nc', 'member': [{'memberId': 'did:ckdr:poigfJXJH4z2WkOjhANmdf8mYUoIz7+6q9/1Gkr6y0KnFA==', 'type': 'unknown', 'faceCID': ['QmQcfk8NL5hqW5mtiNKjZNw98vgU5YswVSioyXLjvj34qy'], 'role': 'admin'}]}
# print(add_member_to_lmdb(MemberPublish))



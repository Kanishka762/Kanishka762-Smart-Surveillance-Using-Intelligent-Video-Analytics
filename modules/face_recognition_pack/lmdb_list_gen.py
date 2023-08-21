from modules.components.load_paths import *
import numpy as np
import cv2 as cv
import lmdb
from io import BytesIO
import io
from PIL import Image
import face_recognition 
import os
from os.path import join, dirname

blacklist_faces = []
blacklist_ids = []
whitelist_faces = []
whitelist_ids = []
unknown_faces = []
unknown_ids = []

# cwd = os.getcwd()
# static_path = join(cwd, 'static')
# lmdb_path = join(static_path, 'lmdb')

#load lmdb
env = lmdb.open(lmdb_path+'/face-detection.lmdb',
                max_dbs=10, map_size=int(100e9))


# Now create subdbs for known and unknown people.
whitelist_db = env.open_db(b'white_list')
blacklist_db = env.open_db(b'black_list')
unknown_db = env.open_db(b'unknown')

def insertWhitelistDb():
    # begin = time.time()
    with env.begin() as txn:
        list1 = list(txn.cursor(db=whitelist_db))
        
    db_count_whitelist = 0
    for key, value in list1:
        #fetch from lmdb
        with env.begin() as txn:
            re_image = txn.get(key, db=whitelist_db)
            
            finalNumpyArray = np.array(Image.open(io.BytesIO(re_image))) 
            image = finalNumpyArray
            try :
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError as e  :
                continue
            whitelist_faces.append(encoding)
            whitelist_ids.append(key.decode())
            db_count_whitelist += 1
            
    # end = time.time()

    # print(f"Total runtime of the program is {end - begin}")
    # print(whitelist_faces)
    print(db_count_whitelist, "total whitelist person")
    return whitelist_faces, whitelist_ids



def insertBlacklistDb():
    # begin = time.time()
    with env.begin() as txn:
        list1 = list(txn.cursor(db=blacklist_db))
        
    db_count_blacklist = 0
    for key, value in list1:
        #fetch from lmdb
        with env.begin() as txn:
            re_image = txn.get(key, db=blacklist_db)
            
            finalNumpyArray = np.array(Image.open(io.BytesIO(re_image))) 
            image = finalNumpyArray
            try :
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError as e  :
                continue
            blacklist_faces.append(encoding)
            blacklist_ids.append(key.decode())
            db_count_blacklist += 1
            
    # end = time.time()
    # print(blacklist_faces)
    # print(f"Total runtime of the program is {end - begin}")
    print(db_count_blacklist, "total blacklist person")
    return blacklist_faces,blacklist_ids



def insertUnknownDb():
    # begin = time.time()
    with env.begin() as txn:
        list1 = list(txn.cursor(db=unknown_db))
        
    db_count_unknown = 0
    for key, value in list1:
        #fetch from lmdb
        with env.begin() as txn:
            re_image = txn.get(key, db=unknown_db)
            
            finalNumpyArray = np.array(Image.open(io.BytesIO(re_image))) 
            image = finalNumpyArray
            try :
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError as e  :
                continue
            unknown_faces.append(encoding)
            unknown_ids.append(key.decode())
            db_count_unknown += 1
            
    # end = time.time()
    # print(blacklist_faces)
    # print(f"Total runtime of the program is {end - begin}")
    print(db_count_unknown, "total unknown person")
    return unknown_faces,unknown_ids
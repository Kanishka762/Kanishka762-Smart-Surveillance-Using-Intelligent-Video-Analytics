from modules.components.load_paths import *
from init import loadLogger

from dotenv import load_dotenv
load_dotenv(dotenv_path)

import ast
import os
import lmdb
import json

track_type = ast.literal_eval(os.getenv("track_type"))

db_env = lmdb.open(f'{lmdb_path}/face-detection', max_dbs=10)
IdLabelInfoDB = db_env.open_db(b'IdLabelInfoDB', create=True)
trackIdMemIdDictDB = db_env.open_db(b'trackIdMemIdDictDB', create=True)
logger = loadLogger()

#crate a class called lmdboperations
class lmdboperations:
    def __init__(self):
        pass
    #create a method called fetchLmdb
    def fetchLMDB(self,db_txn, key):
        try:
            # logger.info("trying updating data in LMDB")
            value = db_txn.get(key.encode())
            if value is None:
                return None
            data = json.loads(value.decode())
            logger.info("updated data in LMDB")
            return data
        except Exception as e:
            logger.error("An error occurred while updating data in LMDB", exc_info=e)

    def insertLMDB(self,db_txn, key,value):
        try:
            logger.info("trying inserting data in LMDB")
            db_txn.put(key.encode(), json.dumps(value).encode())
            logger.info("inserted data in LMDB")
        except Exception as e:
            logger.error("An error occurred while inserting data in LMDB", exc_info=e)




#create a class called classLabels
class classLabels:
    def __init__(self):
        pass

    #create a method called generatelLabelsMeta
    def generateLabelsMeta(self,memID):
        try:
            if memID is None:
                member_track_type = None

            elif memID == '100':
                if memID in track_type:
                    member_track_type = track_type[memID]
            else:
                mem_type = memID[:2]
                if mem_type in track_type:
                    member_track_type = track_type[mem_type]
            return member_track_type
        except Exception as e:
            logger.error("An error occurred while creating member_track_type for live stream labels", exc_info=e)

    def fetch_activity_info(self,detect_type):
        try:
            key = "IdLabelInfo"
            output_dict = {}
            lmdbOps = lmdboperations()
            with db_env.begin(db=IdLabelInfoDB, write=True) as db_txn:
                data = lmdbOps.fetchLMDB(db_txn, key)
            if data is None:
                return None
            for key, value in data.items():
                memID = value['memberID']
                activities = list(set(value['activity']))

                member_track_type = self.generateLabelsMeta(memID)

                sentence = (
                    f"{detect_type} {' '.join(activities)}"
                    if member_track_type is None
                    else f"{member_track_type} {detect_type} {' '.join(activities)}"
                )
                output_dict[key] = sentence

            return output_dict
        except Exception as e:
            logger.error("An error occurred while generating labels with activity and face results for live stream", exc_info=e)
            
    def fetch_member_info(self,detect_type):
        try:
            key = "trackIdMemIdDict"
            output_dict = {}
            lmdbOps = lmdboperations()
            with db_env.begin(db=trackIdMemIdDictDB, write=True) as db_txn:
                data = lmdbOps.fetchLMDB(db_txn, key)
            if data is None:
                return None
            for key, value in data.items():
                memID = (data[key])[-1]

                member_track_type = self.generateLabelsMeta(memID)
                
                sentence = (
                    f"{detect_type}"
                    if member_track_type is None
                    else f"{member_track_type} {detect_type}"
                )
                output_dict[key] = sentence
            return output_dict
        except Exception as e:
            logger.error("An error occurred while generating labels with face results for live stream", exc_info=e)


from modules.face_recognition_pack.facedatainsert_lmdb import add_member_to_lmdb

def load_lmdb_fst(mem_data):
    i = 0
    for each in mem_data:
        i = i+1
        add_member_to_lmdb(each)
        print("inserting ",each)

from modules.face_recognition_pack.lmdb_list_gen import attendance_lmdb_known, attendance_lmdb_unknown
from modules.face_recognition_pack.facedatainsert_lmdb import add_member_to_lmdb


known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []

def load_lmdb_list():

    known_whitelist_faces1, known_whitelist_id1 = attendance_lmdb_known()
    known_blacklist_faces1, known_blacklist_id1 = attendance_lmdb_unknown()
    
    global known_whitelist_faces
    known_whitelist_faces = known_whitelist_faces1

    global known_whitelist_id
    known_whitelist_id = known_whitelist_id1
    
    global known_blacklist_faces
    known_blacklist_faces = known_blacklist_faces1

    global known_blacklist_id
    known_blacklist_id = known_blacklist_id1


def load_lmdb_fst(mem_data):
    i = 0
    for each in mem_data:
        i = i+1
        add_member_to_lmdb(each)
        print("inserting ",each)
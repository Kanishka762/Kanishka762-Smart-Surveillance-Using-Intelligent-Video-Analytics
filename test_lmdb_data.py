
from modules.face_recognition_pack.lmdb_list_gen import insertWhitelistDb, insertBlacklistDb, insertUnknownDb

while True:
    # whitelist_faces1 = []
    # whitelist_id1 = []
    # blacklist_faces1 = []
    # blacklist_id1 = []
    # unknown_faces1 = []
    # unknown_id1 = []
    try:
        whitelist_faces1, whitelist_id1 = insertWhitelistDb()
        blacklist_faces1, blacklist_id1 = insertBlacklistDb()
        unknown_faces1, unknown_id1 = insertUnknownDb()

        print(len(whitelist_faces1),len(blacklist_faces1),len(unknown_faces1))
        print(len(whitelist_id1),len(blacklist_id1),len(unknown_id1))
    except:
        pass

# global whitelist_faces
# # print("before",whitelist_faces)
# whitelist_faces = whitelist_faces1
# # print("after",whitelist_faces)

# global whitelist_ids
# whitelist_ids = whitelist_id1

# global blacklist_faces
# blacklist_faces = blacklist_faces1
# global blacklist_ids
# blacklist_ids = blacklist_id1

# global unknown_faces
# unknown_faces = unknown_faces1
# global unknown_ids
# unknown_ids = unknown_id1
# print(len(whitelist_faces),len(blacklist_faces),len(unknown_faces))

# print(len(whitelist_ids),len(blacklist_ids),len(unknown_ids))
import lmdb
import json
import multiprocessing
import time

db_env = lmdb.open("/home/srihari/deepstreambackend/static/lmdb"+'/face-detection',
                max_dbs=10,
                max_readers=256)
faceDataDictDB = db_env.open_db(b'faceDataDictDB')

def fetchLMDB(db_txn, key):
    value = db_txn.get(key.encode())
    if value is not None:
        data = json.loads(value.decode())
        for category in data:
            if category in []:
                listOfNumpuArray = []
                for encodedData in data[category]:
                    numpyArray = np.array(json.loads(encodedData))
                    # print('numpyArray',type(numpyArray))
                    listOfNumpuArray.append(numpyArray)
                data[category] = listOfNumpuArray

        return data
    else:
        return None


def insertLMDB(db_txn, key,value):
    print("inside with")
    print(value)
    for category in value:
        print(category)
        if category in []:
            convList = []
            for encodings in value[category]:
                print('before encoding',encodings)
                encodings = json.dumps(encodings.tolist())
                print('encodings',type(encodings))
                convList.append(encodings)
            value[category] = convList

    db_txn.put(key.encode(), json.dumps(value).encode())

def insert():
    
    print('insert process started')
    i = 0
    while True:
        time.sleep(5)
        i = i+1223
        try:

            with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
                faceData = {"whitelist_faces":[i],"whitelist_ids":[i],"blacklist_faces":[i],"blacklist_ids":[i],"unknown_faces":[i],"unknown_ids":[i]}
                insertLMDB(db_txn, "faceData", faceData)
            print('Data inserted successfully')
        except Exception as e:
            print(f'Insertion error: {e}')
        # time.sleep(1)  # Adjust the interval as needed

def fetch1():
    while True:
        print('fetch1 process started')

        try:
            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")
            print(faceData)
        except Exception as e:
            print(f'Fetching error: {e}')
        # time.sleep(2)  # Adjust the interval as needed
def fetch2():
    while True:
        print('fetch2 process started')

        try:
            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")
            print(faceData)
        except Exception as e:
            print(f'Fetching error: {e}')
        # time.sleep(2)  # Adjust the interval as needed
def fetch3():
    # print('fetch process started')
    while True:
        print('fetch3 process started')

        try:
            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")
            print(faceData)
        except Exception as e:
            print(f'Fetching error: {e}')
        # time.sleep(2)  # Adjust the interval as needed
def fetch4():
    print('fetch process started')
    while True:
        try:
            print('fetch4 process started')

            with db_env.begin(db=faceDataDictDB) as db_txn:
                faceData = fetchLMDB(db_txn, "faceData")
            print(faceData)
        except Exception as e:
            print(f'Fetching error: {e}')
        # time.sleep(2)  # Adjust the interval as needed


def insert_1():
    print('insert_1 process started')
    # while True:
    try:
        with db_env.begin(db=faceDataDictDB, write=True) as db_txn:
            faceData = {"whitelist_faces":[9],"whitelist_ids":[9],"blacklist_faces":[9],"blacklist_ids":[9],"unknown_faces":[9],"unknown_ids":[9]}      
            insertLMDB(db_txn, "faceData", faceData)
        print('Data inserted successfully (insert_1)')
    except Exception as e:
        print(f'Insertion error (insert_1): {e}')
    # time.sleep(1)  # Adjust the interval as needed

if __name__ == "__main__":
    print('Main process started')



    p2 = multiprocessing.Process(target=fetch1)
    p2.daemon = True
    p2.start()

    p2 = multiprocessing.Process(target=fetch2)
    p2.daemon = True
    p2.start()

    p2 = multiprocessing.Process(target=fetch3)
    p2.daemon = True
    p2.start()

    p2 = multiprocessing.Process(target=fetch4)
    p2.daemon = True
    p2.start()
    time.sleep(5)

    insert_1()
    # p3 = multiprocessing.Process(target=insert_1)
    # p3.daemon = True
    # p3.start()
    
    time.sleep(10)
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")
    print("\n")

    # insert()

    p1 = multiprocessing.Process(target=insert)
    p1.daemon = True
    p1.start()

    # Add some logic here to keep the main process running, e.g., a KeyboardInterrupt handler.
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Main process terminated by user')
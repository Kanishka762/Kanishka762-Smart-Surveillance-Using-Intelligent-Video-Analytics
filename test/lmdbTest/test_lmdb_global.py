import lmdb
import pickle
import time
import threading


# Function to update the LMDB database for a specific dictionary
def update_database(db_txn, key, value):
    db_txn.put(key.encode(), pickle.dumps(value))

# Function to continuously fetch data from a specific LMDB database
def fetch_data(db_txn, key):
    value = db_txn.get(key.encode())
    if value is not None:
        data = pickle.loads(value)
        print(f"Fetched data: {data}")
    else:
        print("Data not found")
    # You can add a sleep here to control the fetching rate

def insert(db1):
    i = 0
    while True:
        i = i +1
        update_data = {}
        with db_env.begin(db=db1, write=True) as txn:
            update_database(txn, "my_key", update_data)
            print("inserted: ",update_data)

        time.sleep(5)

if __name__ == "__main__":
    # Initialize LMDB environment and open databases with increased max_dbs limit
    db_env = lmdb.open("/home/srihari/deepstreambackend/static/lmdb/face-detection.lmdb", max_dbs=10)  # Adjust max_dbs as needed

    # Open databases
    db1 = db_env.open_db(b'db1', create=True)

    threading.Thread(target = insert,args = (db1,)).start()

    
    while True:
        with db_env.begin(db=db1) as txn1:
            fetch_data(txn1, "my_key")
            time.sleep(2)
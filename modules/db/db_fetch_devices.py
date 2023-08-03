from modules.components.load_paths import *
import psycopg2
from pytz import timezone 
from datetime import datetime
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast

# cwd = os.getcwd()
# data_path = join(cwd, 'data')
# dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")
tenant_ids = os.getenv("TENANT_IDS")

print(pg_url, pgdb, pgport, pguser, pgpassword)

def fetch_db():
    # try:
    # Establish a connection to the PostgreSQL database
    connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
    # Create a cursor object
    cursor=connection.cursor()
    query =  '''CREATE TABLE log02082023 PARTITION OF logs
    FOR VALUES FROM ('2023-08-01') TO ('2023-08-02')
        ;'''
    cursor.execute(query)
    connection.commit()
        # print("Selecting rows from device table using cursor.fetchall")
        # device_records = cursor.fetchall()
        # return(device_records)



# device_data = fetch_db()
# print(device_data)
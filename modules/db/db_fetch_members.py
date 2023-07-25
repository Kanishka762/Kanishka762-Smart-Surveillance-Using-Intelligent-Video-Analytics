import psycopg2
from pytz import timezone 
from datetime import datetime
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast

cwd = os.getcwd()
data_path = join(cwd, 'data')
dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")
tenants = ast.literal_eval(os.getenv("TENANT_IDS"))
value_str = ', '.join(f"'{value}'" for value in tenants)

ack = False

def check_null(member_data):
    member_final_list = []
    for each in member_data:
        member = each['member']
        for item in member:
            if((item['memberId'] != None) and (item['faceCID'] != None) and (item['role'] != None)):
                member_final_list.append(each)
    return (member_final_list)

def fetch_db_mem():
    outt = []
    connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
    cursor=connection.cursor()
    cursor.execute(f"""SELECT id, type, "tenantId", track, "firstName", "userId", "blackListed", faceid
    FROM "Member" WHERE id IS NOT NULL AND type IS NOT NULL AND "tenantId" in ({value_str});""")
    members = []
    for row in cursor.fetchall():
        # print(row)
        inn_dict = {}
        member_info={}
        member_info['id'] = row[0]
        member_info["member"] = []
        inn_dict['memberId'] = row[0]
        inn_dict['type'] = row[1]
        inn_dict['faceCID'] = row[7]
        member_info["member"].append(inn_dict)
        outt.append(member_info)
    return outt


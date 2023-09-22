from modules.components.load_paths import *
from init import loadLogger
import psycopg2
from pytz import timezone 
from datetime import datetime
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast

logger = loadLogger()

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
    try:
        logger.info( "checking null in members dictionary")
        member_final_list = []
        for each in member_data:
            member = each['member']
            member_final_list.extend(
                each
                for item in member
                if (
                    (item['memberId'] != None)
                    and (item['faceCID'] != None)
                    and (item['role'] != None)
                )
            )
        return (member_final_list)
    except Exception as e:
        logger.error("An error occured while checking null in members dictionary", exc_info=e)
        

def fetch_db_mem():
    try:
        logger.info( "creating members dictionary")
        outt = []
        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
        cursor=connection.cursor()
        cursor.execute(f"""SELECT id, type, "tenantId", track, "firstName", "userId", "blackListed", faceid
        FROM "Member" WHERE id IS NOT NULL AND type IS NOT NULL AND "tenantId" in ({value_str}) AND track IS NOT NULL AND faceid IS NOT NULL;""")
        members = []
        for row in cursor.fetchall():
            member_info = {'id': row[0], 'updated': False, "member": []}
            inn_dict = {'memberId': row[0], 'type': row[1], 'faceCID': row[7]}
            member_info["member"].append(inn_dict)
            outt.append(member_info)
        logger.info("members data")
        logger.info(outt)
        return outt
    except Exception as e:
        logger.error("An error occured while creating members dictionary", exc_info=e)
        


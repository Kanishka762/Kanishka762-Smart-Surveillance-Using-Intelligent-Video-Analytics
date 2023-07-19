import psycopg2
from pytz import timezone 
from datetime import datetime
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast

dotenv_path = join(dirname(__file__), '.env')
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
    # try:
    # member_info = {}
    outt = []
    # connection = psycopg2.connect(host='216.48.182.5', database='postgres',port='5432',user='postgres',password='Happy@123')
    # connection = psycopg2.connect(host="216.48.182.5", database="postgres", port="8081", user="postgres", password="Happy@123")
    connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
    cursor=connection.cursor()
    
    # cursor.execute("""
    #                 SELECT *
    #                 FROM "Member";
    #             """)
    
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

    #     member = {
    #         "id": row[0],
    #         "member": [],
    #         "type": row[1],
    #         "tenantId": row[2],
    #         "track": row[3],
    #         "firstName": row[4]]),
    #         "blackListed": row[6],
    #         "faceid": row[7]
    #     }
    #     members.append(member)
        
    # connection.commit()
    # print(members),
    #         "userId": str(row[5
    # return members
        
        # query='''SELECT * FROM "Member";'''
        # cursor.execute(query)
        
        # # print("Selecting rows from device table using cursor.fetchall")
        # device_records = cursor.fetchall()
        
        
        
        # for row in device_records:
        #     print(row)
        #     print('\n')


            # inn_dict = {}
            # member_info={}
            
            # member_info['id'] = row[0]
            # member_info["member"] = []
            # inn_dict['memberId'] = row[2]
            # inn_dict['type'] = row[4]
            # inn_dict['faceCID'] = row[28]
            # inn_dict['role'] = row[3]
            # member_info["member"].append(inn_dict)
            # outt.append(member_info)

        # # output = check_null(outt)
        
        # return device_records
    
    # except (Exception, psycopg2.Error) as error:
    #     print("Error while fetching data from PostgreSQL", error)
        
    # finally:
    #     # closing database connection.
    #     if connection:
    #         cursor.close()
    #         connection.close()
    #         print("PostgreSQL connection is closed")


# device_data = fetch_db_mem()
# print(device_data)

# fetch_db_mem()


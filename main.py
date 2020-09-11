import base64
import json
import configparser
import argparse
import os, io
import pickle
import time

import requests
from requests.adapters import HTTPAdapter
import numpy as np
import pandas as pd
# import googleapiclient.errors
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

import db_interface as db
# from model import fema_score

from custom_exceptions import CustomException

config = configparser.ConfigParser()                                     
config.read('config.ini')

MAX_RETRIES = config.get('REQUESTS', 'MAX_RETRIES')
TIMEOUT = config.get('REQUESTS', 'TIMEOUT')
BUILD_COL = config.get('BUILDING_DB', 'BUILDING_COLLECTION')
BUILD_IMG_COL = config.get('BUILDING_DB', 'BUILDING_COLLECTION_IMG')

DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
DATA_FOLDER_PATH = os.path.abspath('../../data/building_images')
USER_DATA_PATH = 'data/user_data'

class ProcessRequest:
    """Send requests to external API's using python's request module."""

    def __init__(self, url: str, req_method: str):
        self.url = url
        self.method = req_method
        self.timeout = TIMEOUT
    
    def send_request(self, params: dict=None, headers: dict=None, payload: dict=None, files: dict=None):
        # files = {'file': open('report.xls', 'rb')}

        try:
            adapter = HTTPAdapter(max_retries=MAX_RETRIES)
            session = requests.Session()
            session.mount('https://', adapter)
            session.mount('http://', adapter)
            
            response = session.request(self.method, self.url, params=params, headers=headers, 
                                                                data=payload, files=files, timeout=self.timeout)
            if response.status_code != 200:                                     
                raise Exception(f'{response.status_code} error: {response.reason}')                                     
            return json.loads(response.content.decode('utf-8'))
        
        except Exception as e:
            raise Exception(e)

class Building:
    
    def __init__(self, role, keys, key_values, max_builds):

        self.role = role
        self.keys = [a.lower() for a in keys]
        self.key_values = [a.lower() for a in key_values]
        # self.key_values = key_values
        print(self.key_values)
        print(self.keys)
        self.max_builds = int(max_builds)

    def retrieve_from_db(self):
        # Moderator allowed to see all
        # User only allowed to see own area and own building
        
        if self.role == 'user': self.check_key_validity()
        db_obj = db.DatabaseInterface(BUILD_COL)
        query = {k:v for k,v in zip(self.keys, self.key_values)}

        result = db_obj.get_by_limit(query, self.max_builds)
        if result == False: raise CustomException('404: No buildings found')
        print(result)
        return result

    def check_key_validity(self):
        # Check if user is allowed to access

        for i in self.keys:
            if i != 'lat_long':
                raise CustomException('401: Unauthorized to access resource')

# class PreTrain:
#     """Process dataset, calculate FEMA score and save building information to database."""
#     def __init__(self):
#         pass

#     def pre_train(self, folder_name):
        
#         try:
#             # folder_data = fema_score.process_folder(folder_name)
#             folder_data = fema_score.process_csv_file(folder_name)
#             res = db.DatabaseInterface('buildings').insert(folder_data)
#             print(f"Pre-training successful")
#             return True, 'Pre-training successful'
#         except Exception as e:
#             print(f"Pre-training failed due to {str(e)}")
#             return False, str(e)
    
#     def parse_folders(self):

#         with open('folders.txt', 'r') as file:
#             content = file.readlines()
#         folders_txt = [x.strip() for x in content] 

#         # Get collection from DB(Contains information of already processed folders)
#         db_obj = db.DatabaseInterface('folders')
#         result = db_obj.get_by_limit()
#         folders_db = [item['name'] for item in result]

#         new_folders = list(set(folders_txt) - set(folders_db))
#         print(f"Folders to pre_train: {new_folders}")
#         for folder in new_folders:
#             success, _ = self.pre_train(folder)
#             if success == True:
#                 result = db_obj.insert([{'name': folder}])
#             pass

        # # Check if new images were added to any folder
        # for folder in folders_db:
        #     folder_path = os.path.join(DATA_FOLDER_PATH, folder)
        #     dir_names=[int(item) for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
        #     print(max(dir_names))

# class GoogleDrive:
#     # Add that folder to folders.txt file
   
#     def __init__(self, folder_name):
#         self.folder_name = folder_name
#         creds = None
    
#         if os.path.exists('drive/token.pickle'):
#             with open('drive/token.pickle', 'rb') as token:
#                 creds = pickle.load(token)

#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     'drive/credentials.json', DRIVE_SCOPES)
#                 creds = flow.run_local_server(port=0)

#             with open('drive/token.pickle', 'wb') as token:
#                 pickle.dump(creds, token)

#         self.service = build('drive', 'v3', credentials=creds)

#     def get_folder_names(self):
        
#         try:
#             if self.folder_name == '' or self.folder_name == ' ': raise Exception('invalid folder')
#             print(f"Downloading data of folder {self.folder_name}")
#             _, main_folder_id = self.file_query(f"name='{self.folder_name}'")
#             folder_path = os.path.join(DATA_FOLDER_PATH, self.folder_name)
#             os.mkdir(folder_path)
#             folder_names, folder_ids = self.file_query(f"'{main_folder_id[0]}' in parents")
            
#             for folder_name, folder_id in zip(folder_names, folder_ids):
#                 os.mkdir(os.path.join(folder_path, folder_name))
#                 file_names, file_ids = self.file_query(f"'{folder_id}' in parents")
#                 for f_name, f_id in zip(file_names, file_ids):
#                     f_name = os.path.join(folder_path, folder_name, f_name)
#                     self.save_file(f_id, f_name)
#                 print(f"Successfully processed folder {folder_name} of {self.folder_name}")
#             self.write_to_folders_txt()
#             return f"Files of {self.folder_name} folder downloaded successfully"

#         except Exception as e:
#             return f"Download of files in {self.folder_name} folder failed due to {str(e)}"
    
#     def download_data_sheet(self):
#         pass


#     def file_query(self, query):

#         names, ids, page_token = [], [], None
#         while True:
#             try:
#                 response = self.service.files().list(q=query, spaces='drive',
#                                                     fields='nextPageToken, files(id, name)',
#                                                     pageToken=page_token).execute()
#                 for file in response.get('files', []):
#                     names.append(file.get('name')) 
#                     ids.append(file.get('id'))
#                 page_token = response.get('nextPageToken', None)
#                 if page_token is None:
#                     break
#             except googleapiclient.errors.HttpError as ge:
#                 raise Exception("File not found")
        
#         return names, ids
    
#     def save_file(self, file_id, file_name):

#         try:
#             request = self.service.files().get_media(fileId=file_id)
#             fh = io.BytesIO()
#             downloader = MediaIoBaseDownload(fh, request)
#             done = False
#             while done is False:
#                 status, done = downloader.next_chunk()
            
#             with open(file_name, 'wb') as f:
#                 f.write(fh.getvalue())
#             fh.close()

#         except Exception as e:
#             raise Exception("Failed to download file")
    
#     def write_to_folders_txt(self):

#         with open('folders.txt', 'a') as f:
#             f.write(self.folder_name + '\n')

class UserData:

    def __init__(self, build_type, area, floors, glass, address, lat_long, images):
        # save images to folder
        # save data to mongodb with address, first get latlong from address
        self.build_type = build_type
        self.floors = int(floors)
        self.address = str(address)
        self.lat_long = self.get_latlong_from_addr()
        self.glass = glass
        self.area = area
        
        self.images = []
        try:
            for img in images:
                if img == b'':  # If image is empty
                    self.images.append(None)
                    raise Exception(f"Image missing for user with address: {self.address}")
                np_arr = np.frombuffer(img, np.uint8)
                img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.images.append(img_np)
            self.save_data()
        
        except Exception as e:
            print(str(e))
    
    def get_latlong_from_addr(self):

        pass

    def save_data(self):
        
        try:
            dir_path = os.path.join(USER_DATA_PATH, self.address)
            os.path.mkdir(dir_path)
            cv2.imwrite(os.path.join(dir_path, '1.jpg'), self.images[0])
            cv2.imwrite(os.path.join(dir_path, '2.jpg'), self.images[1])
            cv2.imwrite(os.path.join(dir_path, '3.jpg'), self.images[2])
            cv2.imwrite(os.path.join(dir_path, 'marked.jpg'), self.images[3])
            cv2.imwrite(os.path.join(dir_path, 'static.jpg'), self.images[4])

            # Save other data to csv file
            user_data_list = [self.area, self.address, self.lat_long, 
                            self.build_type, self.floors, self.glass]
            user_data_df = pd.DataFrame(user_data_list)
            csv_file_path = os.path.join(USER_DATA_PATH, 'user_data.csv')
            user_data_df.to_csv(csv_file_path, mode='a', header=False, encoding='utf-8', index=False)
        
        except Exception as e:
            print(f"Data saving of user with address: {self.address} failed due to {str(e)}")


def return_project_info():

    ret_dict = {}
    db_obj = db.DatabaseInterface('information')
    result = db_obj.get_by_limit()
    for row in result:
        # k,v = list(row.items())[0]
        # ret_dict[k] = v
        ret_dict = {**ret_dict, **row}
    return ret_dict

if __name__ == "__main__": 
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c','--command_type', type=str, required=True)
        parser.add_argument('-f','--folder_name', type=str)
        args = parser.parse_args()
        print(f"Command name: {args.command_type}")

        if args.command_type == 'pre_train':
            print(PreTrain().parse_folders())
        elif args.command_type == 'download_data':
            if args.folder_name is None: raise Exception('No folder name given')
            print(GoogleDrive(args.folder_name).get_folder_names())
        elif args.command_type == 'get_proj_info':
            print(return_project_info())
        else:
            raise Exception("Invalid command")
    
    except Exception as e:
        print(e)

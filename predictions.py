import time, os

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import OneHotEncoder
import joblib

# from model import fema_score as fem
import fema_score as fem
from custom_exceptions import CustomException

dims = 128
dims_half = 128 // 2

soil_info = {'dhanmondi': 'D', 'badda': 'D', 'jatrabari': 'D', 'khilgaon': 'D', 'banani': 'D',
             'uttara': 'D','lalmatia': 'D', 'mohakhali': 'E' , 'niketan': 'E', 'aftabnagar': 'E',
             'panthapath': 'D'}
model = load_model('final_model.h5')
enc = joblib.load('encoder.joblib')

class Predict():
    """Predict objects in an image. Image input as bytes or numpy array."""

    def __init__(self, images, build_type, floors, lat_long, glass, area):
        
        self.build_type = build_type
        self.floors = int(floors)
        self.lat_long = lat_long
        self.glass = glass
        self.area = area
        self.soil_type = self.det_soil_type()
        
        self.images = []
        for img in images:
            if img == b'':  # If image is empty
                self.images.append(None)
                continue
            np_arr = np.frombuffer(img, np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.images.append(img_np)
        
        self.img_one = cv2.resize(self.images[0], (dims_half, dims))
        # self.img_two = cv2.resize(self.images[1], (dims_half, dims))
        # self.img_three = cv2.resize(self.images[2], (dims_half, dims))
        self.img_marked_resized = cv2.resize(self.images[3], (dims_half, dims))
        self.img_marked = self.images[3]
        self.img_static = self.images[4]
    
    def det_soil_type(self):
        # Save soil info in DB and retrieve from there
        try:
            return soil_info[self.area.lower()]
        except:
            return 'D'
            # raise Exception(f"{self.area} is an invalid area.")

    def calc_fema_score(self):
        try:
            fema_obj = fem.FEMA(self.img_marked, self.img_static, self.build_type, self.floors, 
                                self.lat_long, self.glass, self.soil_type, self.area)
            result = fema_obj.calc_fema_score()
            if result['success'] == True:
                self.ver_irrg = result['vertical_irregularity']
                self.plan_irrg = result['plan_irregularity']
                self.pounding = result['pounding']
                self.struct_eval = result['structural_evaluation']
                return {'ver_irrg': self.ver_irrg, 'plan_irrg': self.plan_irrg,
                        'pounding': self.pounding}
            else:
                raise CustomException('500:FEMA Score not calculated')
        
        except CustomException as ce:
            raise CustomException(ce)
        
        except Exception as e:
            raise Exception(e)
    
    def pred_class(self):
        try:
            self.process_data()
            self.preds = model.predict([self.X_enc, self.outputImg])
            self.classes = np.where(self.preds > 0.5, 1, 0)
            self.conf_score = self.preds.tolist()[0][1]
            self.pred_class = True if self.classes[0][0] == 0 else False
            return self.pred_class, self.conf_score
        except Exception as e:
            raise Exception(e)
    
    def process_data(self):
        
        data = {'soil_type': self.soil_type, 'occupancy': self.build_type, 
                'storey': self.floors, 'glass': self.glass}
        df = pd.DataFrame(data, columns = data.keys(), index=range(1))
        X = df.values
        
        self.X_enc = enc.transform(X).toarray()

        self.outputImg = np.zeros((dims, dims, 3), dtype="uint8")
        self.outputImg[0:dims, 0:dims_half] = self.img_one
        self.outputImg[0:dims, dims_half:dims] = self.img_marked_resized
        self.outputImg = np.array([self.outputImg])
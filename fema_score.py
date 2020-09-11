"""Proposed Baseline system determined using FEMA version 3"""

import math
import configparser
from itertools import chain
import copy
import time
import os

import cv2
import imutils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# from model.process_dataset import Data
# from ubera import interfaces

cur_dir = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()                                     
config.read('FEMA.ini')

data_path = os.path.abspath('D:/UBERA/data/building_images')

class FemaException(Exception):
    pass

class FEMA:
    """Calculate FEMA score of a building."""
    
    def __init__(self,marked_path,static_path,building_type,num_storey,lat_long,glass_facade,
                 soil_type, area):
        self.marked_img_path = marked_path
        self.static_img_path = static_path
        self.build_type = building_type
        self.num_storey = num_storey
        self.lat_long = lat_long
        self.glass = glass_facade.lower()
        self.soil_type = soil_type.lower()
        self.area = area.lower()
        self.dist = lambda a,b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        self.grad = lambda l: (l[3]-l[1])/(l[2]-l[0])
    
    def calc_fema_score(self):
        """Assuming moderate seismic region and building type of C1."""
        self.vertical_irregularity = None
        try:
            base = float(config.get('SCORES', 'base'))
            min_score = float(config.get('SCORES', 'min'))

            self.calc_pounding()
            sev_ver_irrg, mod_ver_irrg = 0, 0
            
            self.ver_irrg()
            if self.vertical_irregularity == 'severe':
                sev_ver_irrg = float(config.get('SCORES', 'sev_ver_irrg'))
            elif self.vertical_irregularity == 'moderate':
                mod_ver_irrg = float(config.get('SCORES', 'mod_ver_irrg'))
            
            self.plan_irrg()
            if self.plan_irregularity is None:
                plan_irrg_score = 0
            else:
                plan_irrg_score = float(config.get('SCORES', 'plan_irrg'))
            
            soil_score = self.get_soil_score()
            
            self.final_score = base + sev_ver_irrg + mod_ver_irrg + plan_irrg_score + soil_score
            self.eval_score = min_score if self.final_score < min_score else self.final_score
            struct_eval = self.structural_eval()

            return {'success': True, 'final_score': round(self.final_score, 4),
                    'vertical_irregularity': str(self.vertical_irregularity), 
                    'plan_irregularity': str(bool(plan_irrg_score)), 'soil_score': bool(soil_score), 
                    'pounding': str(self.pounding['result']), 'structural_evaluation': struct_eval}

        except Exception as e:
            # print(e)
            return {'success': False, 'msg': str(e)}

    def get_soil_score(self):

        if self.soil_type == 'a' or self.soil_type == 'b':
            return float(config.get('SCORES', 'soil_a_b'))
        if self.soil_type == 'e':
            if self.num_storey > 3: return float(config.get('SCORES', 'soil_e_high_rise'))
            return float(config.get('SCORES', 'soil_e_low_rise'))
        return 0

    def structural_eval(self):
        """Return if structural evaluation is required based on scores and pounding"""
        cut_off = float(config.get('SCORES', 'cut_off'))
        pounding_status = self.pounding['result']
        if (self.eval_score < cut_off) or (pounding_status):
            return True
        return False


    ############## Pounding ##########################

    def calc_pounding(self):
        """Returns True if pounding is present. False if it isn't. None if there is error.
        Result returned in dictionary, 'result' key has status. Details in other keys.
        'msg' key contains error only if an error occured."""
        # min gap of 0.0127 m/story, for 6-story: gap of 0.0762m AND
        # Adjacent building 2 or more stories taller
        # Floors separated vertically by more than 0.6096m
        # Building at end of block
        
        self.pounding = {'result': None, 'gap': None, 'storey': None, 'floor_height': None}
        self.left, self.right = [True]*2
        try:
            self.get_boundaries() 
            self.pixel_per_metre()
            self.adjacent_buildings()
            min_gap_allowed = (0.0127 * self.num_storey) * self.pixel_per_metre
            self.calc_gap()
            
            gap_poundings = [True for v in self.gaps.values() if v < min_gap_allowed]
            self.gap_pounding = True if True in gap_poundings else False
            
            storey_poundings = [True for v in self.adj_build_storey.values() if v-self.num_storey > 1]
            self.storey_pounding = True if True in storey_poundings else False

            floor_height_poundings = [True for v in self.adj_build_height.values() if v-self.build_height_m > 0.6096]
            self.floor_height_pounding = True if True in floor_height_poundings else False

            pounding_res = self.gap_pounding and (self.storey_pounding or self.floor_height_pounding)
            self.pounding['result'] = True if pounding_res == True else False
            self.pounding['gap'] = self.gap_pounding 
            self.pounding['storey'] = self.storey_pounding
            self.pounding['floor_height'] = self.floor_height_pounding
        
        except FemaException as fe:
            # print(fe)
            if str(fe) == 'Boundaries missing':
                self.pounding['msg'] = str(fe)

        except Exception as e:
            # print(e)
            self.pounding['msg'] = str(e)

    def get_boundaries(self):
        """Get boundaries or outlines of a building."""
        
        if not isinstance(self.marked_img_path, np.ndarray):
            marked_img = cv2.imread(self.marked_img_path)
        # print(marked_img)
        marked_hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)

        lower_range = np.array([0,0,0])       
        upper_range = np.array([20,255,25])
        # Get boundaries of building which is marked by black lines
        self.boundaries = cv2.inRange(marked_hsv, lower_range, upper_range)

        self.get_lines()    # Get vertical and horizontal lines
        top,bottom,right,left = self.process_lines()

        self.right_clus = [self.cluster_lines(region, n_clus=1) for region in right]
        self.left_clus = [self.cluster_lines(region, n_clus=1) for region in left]
        self.top_clus = self.cluster_lines(top[0], n_clus=1)
        self.bottom_clus = self.cluster_lines(bottom[0], n_clus=1)

    def get_lines(self):
        """Get all lines using Hough Transform, Line Detection."""

        # auto_edge = cv2.dilate(self.boundaries, None, iterations=1)
        # auto_edge = cv2.erode(self.boundaries, None, iterations=1)
        lines = cv2.HoughLinesP(self.boundaries,1,np.pi/180,100,minLineLength=100,maxLineGap=200)
        self.ver_lines, self.hor_lines = [], []
        for line in lines:
            x1,y1,x2,y2 = line[0]
            y_diff, x_diff = (y2-y1), (x2-x1)
            x1,y1,x2,y2 = (a.item() for a in (x1,y1,x2,y2))

            if x_diff == 0: 
                self.ver_lines.append([x1,y1,x2,y2])
                continue
            
            m = round(y_diff/x_diff)          # Find gradient
            if m != 0:  # Vertical lines
                self.ver_lines.append([x1,y1,x2,y2])
            else: # Horizontal lines
                self.hor_lines.append([x1,y1,x2,y2])
        
        self.ver_lines.sort(key=lambda x:x[0])
        self.hor_lines.sort(key=lambda x:x[1])

    def process_lines(self):
        """Partition into top,bottom,left,right. Average each of those sections into 1 line."""
        left, right, top, bottom = ([] for i in range(4))
        mid_point_x, mid_point_y = (self.boundaries.shape[1])//2, (self.boundaries.shape[0])//2
        for line in self.ver_lines:
            x1,y1,x2,y2 = line
            if x1 < mid_point_x or x2 < mid_point_x:  
                left.append([x1,y1,x2,y2])
            else:
                right.append([x1,y1,x2,y2])
        
        for line in self.hor_lines:
            x1,y1,x2,y2 = line
            if y1 > mid_point_y or y2 > mid_point_y:
                top.append([x1,y1,x2,y2])
            else:
                bottom.append([x1,y1,x2,y2])

        if all(map(len, [right,left,top,bottom])) == False: 
            raise FemaException('Boundaries missing')

        right_avg, left_avg= self.similar_lines(right, 2), self.similar_lines(left, 2)
        top_avg, bottom_avg= self.similar_lines(top, 1), self.similar_lines(bottom, 1)

        if all(map(len, [right_avg,left_avg,top_avg,bottom_avg])) == False: 
            raise FemaException('Boundaries missing')
        if len(right_avg) == 1: self.right = False
        if len(left_avg) == 1: self.left = False

        return top_avg,bottom_avg,right_avg,left_avg

    def similar_lines(self, lines, n_lines, max_angle=1, max_dist=10):
        """Group lines that have angle less than max_angle between them.
           Or minimum distance between them is less than max_dist."""
        def line_angle(line1, line2):
            m1, m2 = self.grad(line1), self.grad(line2)
            angle = np.arctan(abs((m2-m1)/(1+m1*m2)))
            return math.degrees(angle)
        
        result = []
        for i, l1 in enumerate(lines):
            res = set()
            for j in range(i+1, len(lines)):
                l2 = lines[j]
                angle = line_angle(l1,l2)
                line_dist = self.calc_distance(l1,l2)
                if angle < max_angle or line_dist < max_dist:
                    res.update([i,j])
            if len(res) != 0: result.append(list(res))
        
        similar_line_idx = []
        while len(result)>0:
            first, *rest = result
            first = set(first)

            lf = -1
            while len(first)>lf:
                lf,rest2 = len(first), []
                for r in rest:
                    if len(first.intersection(set(r)))>0:
                        first |= set(r)
                    else:
                        rest2.append(r)     
                rest = rest2
            similar_line_idx.append(list(first))
            result = rest
        lines = np.array(lines)
        sim_lines = [lines[idxs].tolist() for idxs in similar_line_idx]
        return sim_lines[:n_lines]

    def cluster_lines(self, lines, n_clus):
        """Clustering similar lines together. Centre of the cluster is considered."""

        X = np.array(lines)
        kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(X)
        centres = kmeans.cluster_centers_.astype(int)
        centres = np.squeeze(centres)
        return centres.tolist()
        # return kmeans.labels_

    def pixel_per_metre(self):
        """Calculate conversion between pixels and metre."""
        self.build_height_m = float(config.get('HEIGHTS', self.build_type))*self.num_storey
        build_height_pixel = self.calc_distance(self.top_clus, self.bottom_clus)
        self.pixel_per_metre = build_height_pixel/self.build_height_m
        
    def adjacent_buildings(self):
        """Find number of stories of adjacent buildings"""

        # Adjacent building, number of stories. Assume n/a building type for adjacents.
        build_height_na = float(config.get('HEIGHTS', 'n/a'))
        self.adj_build_height, self.adj_build_storey = {}, {}
        
        if self.left == True:
            left_build_height = self.dist(self.left_clus[0][0:2],self.left_clus[0][2:4])
            left_build_height = left_build_height / self.pixel_per_metre
            self.adj_build_height['left'] = left_build_height
            left_build_storey = left_build_height // build_height_na
            self.adj_build_storey['left'] = int(left_build_storey)
        
        if self.right == True:
            right_build_height = self.dist(self.right_clus[1][0:2],self.right_clus[1][2:4])
            right_build_height = right_build_height / self.pixel_per_metre
            self.adj_build_height['right'] = right_build_height
            right_build_storey = right_build_height // build_height_na
            self.adj_build_storey['right'] = int(right_build_storey)

    def calc_distance(self, line1, line2):
        """Calculate distance between two buildings/lines in pixels."""
        # Here each x,y combination is a point on a line
        xy1,xy2 = line1[:2], line1[2:4]
        xy3,xy4 = line2[:2], line2[2:4]
        
        distances = []
        for xy in (xy1, xy2):       # Use itertools.combination
            distances.append(self.dist(xy,xy3))
            distances.append(self.dist(xy,xy4))
        
        return min(distances)

    def display_lines(self, lines):
        result = np.zeros(self.boundaries.shape)
        print(self.boundaries.shape)
        for i,line in enumerate(lines):
            x1,y1,x2,y2 = line
            cv2.line(result,(x1,y1),(x2,y2),(255,255,255),2)

        # cv2.imshow('result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result

    def calc_gap(self):
        """Calculate gaps between the vertical lines i.e. adjacent buildings."""
        
        left_gap_m, right_gap_m = 99999, 99999
        if self.right == True:
            right_gap_pixel = self.calc_distance(self.right_clus[0], self.right_clus[1])
            right_gap_m = right_gap_pixel / self.pixel_per_metre
        
        if self.left == True:
            left_gap_pixel = self.calc_distance(self.left_clus[0], self.left_clus[1])
            left_gap_m = left_gap_pixel / self.pixel_per_metre
        
        self.gaps = {'left': left_gap_m, 'right': right_gap_m}

    ########## Plan Irregularity ################
    def plan_irrg(self):
        """Determine if plan irregularity exists."""
        self.build_shape, self.plan_irregularity = None, None
        try:
            if not isinstance(self.static_img_path, np.ndarray):
                img = cv2.imread(self.static_img_path)
            
            self.hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.get_static_builds()
            self.get_shape()

            self.torsion = True if self.glass == 'yes' else False
            if self.torsion or self.build_shape == 'irregular':
                self.plan_irregularity = True
        
        except Exception as e:
            pass
        
    def get_static_builds(self):

        # Get Yellow buildings
        lower_range = np.array([12, 14, 255])       
        upper_range = np.array([25,200,255])
        yellow = cv2.inRange(self.hsv_img, lower_range, upper_range)

        # Get Grey buildings
        lower_range = np.array([90, 2, 240])       
        upper_range = np.array([110,7,247])
        grey = cv2.inRange(self.hsv_img, lower_range, upper_range)

        # Combine both, binary image created
        self.static_shape = cv2.bitwise_or(yellow,grey, mask=None)
        # cv2.imshow("Static shape", self.static_shape)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()       
        
    def get_shape(self):
        """Get contours of the image.Then determine if shape is regular.
        Regular shape is rectangular or square."""
        _,contours,_ = cv2.findContours(self.static_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.pos_contours = []
        for cnt in contours:
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)  # Approximate number of sides in shapes
            poly_len = len(approx)
            # print(poly_len)
            if poly_len > 3:
                _, _, w, h = cv2.boundingRect(approx)
                ar = w / float(h)                                   # Aspect ratio
                
                if poly_len == 4 and ar < 1.20:
                    self.pos_contours.append(cnt)
                elif poly_len > 4 and ar < 1.20:
                    self.build_shape = 'irregular'  # has plan irregularity
                    break
        
        if len(self.pos_contours): self.build_shape = 'regular'


    ################ Vertical irregularity #####################
    # One story is taller than all others: Found by comparing height from image and predicted height using building code
    # Severe irregularity if:
    # 1) Ground floor has parking or display showrooms
    # 2) Bottom story taller than other stories or has short column
    # 3) Story height is irregular or unequal
    # Moderate irregularity if:
    # 1) Sloping site: Large difference in gradient between top & bottom of building. Or
    #                  large height difference between the sides of building
    # 2) Split levels: Large height difference between the sides of building

    def ver_irrg(self, max_len_diff=20, max_grad_diff=5):
        """Determine if vertical irregularity exists."""
        severe_ver_irrg, moderate_ver_irrg = False, False
        try:
            if self.pounding['result'] is not None: 
                left_bound = self.left_clus[0] if self.left == False else self.left_clus[1]
                right_bound = self.right_clus[0]
                right_len = self.dist(right_bound[0:2], right_bound[2:4])
                left_len = self.dist(left_bound[0:2], left_bound[2:4])
                len_diff = abs(right_len - left_len)

                top_bound, bottom_bound = self.top_clus, self.bottom_clus
                top_m, bottom_m = self.grad(top_bound), self.grad(bottom_bound)
                m_diff = abs(top_m - bottom_m)

                if (len_diff > max_len_diff) or (m_diff > max_grad_diff):
                    moderate_ver_irrg = True
            
            if self.glass == 'yes':
                severe_ver_irrg = True
            
            if severe_ver_irrg:
                self.vertical_irregularity = 'severe'
            elif severe_ver_irrg == False and moderate_ver_irrg == True:
                self.vertical_irregularity = 'moderate'
            else:
                self.vertical_irregularity = None

        except Exception as e:
            pass
            # print('Vertical Irregularity determination failed')

# def process_folder(folder_name):
    
#     try:
#         print(folder_name)
#         start = time.time()
#         data_obj = Data(folder_name)
#         data_obj.load_build_attr()
#         df = data_obj.data_frame

#         for index, row in df.iterrows():
#             f = FEMA(row['marked'], row['static'], row['occupancy'], row['storey'],
#                     row['lat_long'], row['glass'], row['soil_type'], row['area'])
#             result = f.calc_fema_score()

#             if result['success'] == False: df.drop(index)
#             df.loc[index, 'vertical_irregularity'] = str(result['vertical_irregularity']).lower()
#             df.loc[index, 'plan_irregularity'] = str(result['plan_irregularity']).lower()
#             df.loc[index, 'pounding'] = str(result['pounding']).lower()
#             df.loc[index, 'structural_evaluation'] = str(result['structural_evaluation']).lower()
#             df.loc[index, 'occupancy'] = str(row['occupancy']).lower()
#             df.loc[index, 'storey'] = str(row['storey'])
#             df.loc[index, 'area'] = str(row['area']).lower()
        
#         df.drop(['marked', 'static'], axis=1, inplace=True)
#         csv_file_path = os.path.join(data_path, folder_name, folder_name+'_with_fema.csv')
#         df.to_csv(csv_file_path, encoding='utf-8', index=False)
#         print(time.time()-start)

#         return df.to_dict(orient='records')

#     except Exception as e:
#         raise Exception(str(e))

# def process_csv_file(folder_name):

#     print(folder_name)
#     folder_path = os.path.join(data_path, folder_name)
#     csv_path = os.path.join(folder_path, folder_name+'_with_fema.csv')
#     df = pd.read_csv(csv_path)

#     for index,row in df.iterrows():

#         df.loc[index, 'vertical_irregularity'] = str(row['vertical_irregularity']).lower()
#         df.loc[index, 'plan_irregularity'] = str(row['plan_irregularity']).lower()
#         df.loc[index, 'pounding'] = str(row['pounding']).lower()
#         df.loc[index, 'structural_evaluation'] = str(row['structural_evaluation']).lower()
#         df.loc[index, 'occupancy'] = str(row['occupancy']).lower()
#         df.loc[index, 'storey'] = str(row['storey'])
#         df.loc[index, 'area'] = str(row['area']).lower()

#     return df.to_dict(orient='records')



if __name__ == "__main__":
    # dict_result = process_folder('samples')
    # print(dict_result)

    # f = FEMA('D:/UBERA/data/building_images/mirpur/22/marked.png', 
    #          'D:/UBERA/data/building_images/mirpur/22/static.png', 'mixed', 6,
    #                 '23.788215, 90.347885', 'Yes', 'D', 'mirpur')
    # result = f.calc_fema_score()
    # print(result)

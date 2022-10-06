import cv2
import numpy as np
import math
import threading
import time
import json
class imageprocessing:
###############################################################################
# 程式初始化
###############################################################################
    def __init__(self,url = 0, capture = True, ui = False):
        
        if capture:
            print("open capture")
            self.capture = cv2.VideoCapture(url)
            self.isstop = False
        
        self.model = "nothing"
        
        self.x_cm = 0
        self.y_cm = 0
        self.status = 0
        self.real_high = 0
        
        self.pixel_cm = self.getpixelcm(self.real_high)
        
        self.kernel_size = (1,3,5,7,9,11,13,15,17,19)
        
        self.ui = ui
        
        self.landing_open = False
        self.putin_open = False
        self.red_light_open = False
        self.takeoff_open = False
        self.takeoff_open_2 = False
        
        self.red_first = False
        self.red_time = 0
        
        self.segmentation1 = 160
        self.segmentation2 = 160
        
        try:
            with open("output.json") as f:
                p = json.load(f)
                
                self.area_line_follower = p['area_line_follower']
                self.h_l = p['h_l']
                self.s_l = p['s_l']
                self.v_l = p['v_l']
                self.h_h = p['h_h']
                self.s_h = p['s_h']
                self.v_h = p['v_h']
                self.mid_k = p["mid_k"]
                self.Gaus_k = p["Gausk"]
                self.kernel = p["kernel"]
                self.e1 = p["e1"]
                self.d1 = p["d1"]
                self.e2 = p["e2"]
                self.d2 = p["d2"]

                self.area_landing = p['area_landing']
                self.h_l_r = p['h_l_r_landing']
                self.s_l_r = p['s_l_r_landing']
                self.v_l_r = p['v_l_r_landing']
                self.h_h_r = p['h_h_r_landing']
                self.s_h_r = p['s_h_r_landing']
                self.v_h_r = p['v_h_r_landing']
                
                self.h_l_g = p['h_l_g_landing']
                self.s_l_g = p['s_l_g_landing']
                self.v_l_g = p['v_l_g_landing']
                self.h_h_g = p['h_h_g_landing']
                self.s_h_g = p['s_h_g_landing']
                self.v_h_g = p['v_h_g_landing']
                
                self.h_l_puting = p['h_l_puting']
                self.s_l_puting = p['s_l_puting']
                self.v_l_puting = p['v_l_puting']
                self.h_h_puting = p['h_h_puting']
                self.s_h_puting = p['s_h_puting']
                self.v_h_puting = p['v_h_puting']
                
                self.h_l_red_light = p['h_l_red_light']
                self.s_l_red_light = p['s_l_red_light']
                self.v_l_red_light = p['v_l_red_light']
                self.h_h_red_light = p['h_h_red_light']
                self.s_h_red_light = p['s_h_red_light']
                self.v_h_red_light = p['v_h_red_light']
                    
            print("open json")
            
        except: 
            
            self.area_line_follower = 1000
            self.h_l = 0
            self.s_l = 0
            self.v_l = 0
            self.h_h = 180
            self.s_h = 255
            self.v_h = 100
            
            self.mid_k = 5
            self.Gaus_k = 5
            self.kernel = 5
            self.e1 = 0
            self.d1 = 0 
            self.e2 = 0
            self.d2 = 0
            
            self.area_landing = 10000
            self.h_l_r = 156
            self.s_l_r = 43
            self.v_l_r = 46
            self.h_h_r = 180
            self.s_h_r = 255
            self.v_h_r = 255
            
            self.h_l_g = 35
            self.s_l_g = 43
            self.v_l_g = 46
            self.h_h_g = 77
            self.s_h_g = 255
            self.v_h_g = 255
            
            self.h_l_puting = 100
            self.s_l_puting = 43
            self.v_l_puting = 46
            self.h_h_puting = 124
            self.s_h_puting = 255
            self.v_h_puting = 255
            
            self.h_l_red_light = 156
            self.s_l_red_light = 43
            self.v_l_red_light = 46
            self.h_h_red_light = 180
            self.s_h_red_light = 255
            self.v_h_red_light = 255
            
        if ui :

            cv2.namedWindow('line_follower')
            cv2.resizeWindow('line_follower', 300, 600)
            cv2.createTrackbar('Hue Min', 'line_follower', self.h_l, 180, self.nothing)
            cv2.createTrackbar('Hue Max', 'line_follower', self.h_h, 180, self.nothing)
            cv2.createTrackbar('Sat Min', 'line_follower', self.s_l, 255, self.nothing)
            cv2.createTrackbar('Sat Max', 'line_follower', self.s_h, 255, self.nothing)
            cv2.createTrackbar('Val Min', 'line_follower', self.v_l, 255, self.nothing)
            cv2.createTrackbar('Val Max', 'line_follower', self.v_h, 255, self.nothing)
            cv2.createTrackbar('Median', 'line_follower', self.mid_k, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Gaussian', 'line_follower', self.Gaus_k, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Kernel', 'line_follower', self.kernel, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Erode 1', 'line_follower', self.e1, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Dilate 1', 'line_follower', self.d1, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Erode 2', 'line_follower', self.e2, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Dilate 2', 'line_follower', self.d2, len(self.kernel_size)-1, self.nothing)
            cv2.createTrackbar('Area Min', 'line_follower', self.area_line_follower, 50000, self.nothing)
            cv2.createTrackbar('segmentation1', 'line_follower',self.segmentation1, 200, self.nothing )
            cv2.createTrackbar('segmentation2', 'line_follower',self.segmentation2, 200, self.nothing )

            cv2.namedWindow('landing')
            cv2.resizeWindow('landing', 300, 600)
            cv2.createTrackbar('Hue Min r', 'landing', self.h_l_r, 180, self.nothing)
            cv2.createTrackbar('Hue Max r', 'landing', self.h_h_r, 180, self.nothing)
            cv2.createTrackbar('Sat Min r', 'landing', self.s_l_r, 255, self.nothing)
            cv2.createTrackbar('Sat Max r', 'landing', self.s_h_r, 255, self.nothing)
            cv2.createTrackbar('Val Min r', 'landing', self.v_l_r, 255, self.nothing)
            cv2.createTrackbar('Val Max r', 'landing', self.v_h_r, 255, self.nothing)
            cv2.createTrackbar('Hue Min g', 'landing', self.h_l_g, 180, self.nothing)
            cv2.createTrackbar('Hue Max g', 'landing', self.h_h_g, 180, self.nothing)
            cv2.createTrackbar('Sat Min g', 'landing', self.s_l_g, 255, self.nothing)
            cv2.createTrackbar('Sat Max g', 'landing', self.s_h_g, 255, self.nothing)
            cv2.createTrackbar('Val Min g', 'landing', self.v_l_g, 255,self.nothing)
            cv2.createTrackbar('Val Max g', 'landing', self.v_h_g, 255, self.nothing)
            cv2.createTrackbar('Area Min', 'landing', self.area_landing, 50000, self.nothing)
            
            cv2.namedWindow('puting')
            cv2.resizeWindow('puting', 300, 600)
            cv2.createTrackbar('Hue Min puting', 'puting', self.h_l_puting, 180, self.nothing)
            cv2.createTrackbar('Hue Max puting', 'puting', self.h_h_puting, 180, self.nothing)
            cv2.createTrackbar('Sat Min puting', 'puting', self.s_l_puting, 255, self.nothing)
            cv2.createTrackbar('Sat Max puting', 'puting', self.s_h_puting, 255, self.nothing)
            cv2.createTrackbar('Val Min puting', 'puting', self.v_l_puting, 255, self.nothing)
            cv2.createTrackbar('Val Max puting', 'puting', self.v_h_puting, 255, self.nothing)
            
            cv2.namedWindow('red_light')
            cv2.resizeWindow('red_light', 300, 600)
            cv2.createTrackbar('Hue Min red_light', 'red_light', self.h_l_red_light, 180, self.nothing)
            cv2.createTrackbar('Hue Max red_light', 'red_light', self.h_h_red_light, 180, self.nothing)
            cv2.createTrackbar('Sat Min red_light', 'red_light', self.s_l_red_light, 255, self.nothing)
            cv2.createTrackbar('Sat Max red_light', 'red_light', self.s_h_red_light, 255, self.nothing)
            cv2.createTrackbar('Val Min red_light', 'red_light', self.v_l_red_light, 255, self.nothing)
            cv2.createTrackbar('Val Max red_light', 'red_light', self.v_h_red_light, 255, self.nothing)
            
            print("open ui")
###############################################################################
# 判斷
###############################################################################
    def judge(self, img):
        
        
        
        self.line_follower_found_1 = False
        self.line_follower_found_2 = False
        self.line_follower_found_3 = False
        
        self.red_light_found = False
        self.putin_found = False
        self.landing_found = False
        self.line_follower_found = False
        self.takeoff_found = False
        self.takeoff_found_2 = False
        
        self.seting()
        self.pixel_cm = self.getpixelcm(self.real_high)
        
        self.status2 = 0
        
        if self.takeoff_open_2:
            [x6,y6,s6] = self.takeoff_2(img.copy())
        
        if self.takeoff_open:
            [x5,y5,s5] = self.takeoff(img.copy())
        
        if self.red_light_open:
            [x4,y4,s4] = self.red_light(img.copy())
        
        if self.putin_open :
            [x3,y3,s3] = self.putin(img.copy()) 
            
        if self.landing_open:
            [x2,y2,s2] = self.landing(img.copy())
        
        img = self.img_resize(img)
        
        image = np.empty((480,640,3), dtype ="uint8")
        
        img1 = img[0:self.segmentation1+1,:,:].copy()
        img2 = img[self.segmentation1+1:480-self.segmentation2-1,:,:].copy()
        img3 = img[480-self.segmentation2-1:480,:,:].copy()
        
        [x1_1,y1_1,s1_1,img1_1] = self.line_follower_1(img1)
        [x1_2,y1_2,s1_2,img1_2] = self.line_follower_2(img2)
        [x1_3,y1_3,s1_3,img1_3] = self.line_follower_3(img3)
        
        image[0:self.segmentation1+1,:,:] = img1_1
        image[self.segmentation1+1:480-self.segmentation2-1,:,:] = img1_2
        image[480-self.segmentation2-1:480,:,:] = img1_3
        
        
        state = 0
        
        if self.line_follower_found_3:
            state += 4
        
        if self.line_follower_found_2:
            state += 2
        
        if self.line_follower_found_1:
            state += 1
        
        if state > 0:
            self.line_follower_found = True
            
        if self.takeoff_found_2 :
           self.model = "takeoff_2"
           [self.x_cm, self.y_cm, self.status] = [x6,y6,0]
           
        elif self.takeoff_found :
           self.model = "takeoff"
           [self.x_cm, self.y_cm, self.status] = [x5,y5,0]
        
        elif self.red_light_found:
            
            self.red_time += 1
            if self.red_time > 20:
                self.red_first = True
                
            self.model = "red_light"
            [self.x_cm, self.y_cm, self.status] = [x4,y4,0]
            
        elif self.putin_found:
            self.model = "putin"
            [self.x_cm, self.y_cm, self.status] = [x3,y3,0]
            
        elif self.landing_found:
            self.model = "landing"
            [self.x_cm, self.y_cm, self.status] = [x2,y2,0]

        elif self.line_follower_found:
            self.model = "line_follower"
#==============================================================================            
            if state == 1:
                [self.x_cm, self.y_cm, self.status] = [x1_1, y1_1, s1_1]
                self.status2 = s1_1
                
            elif state == 2:
                [self.x_cm, self.y_cm, self.status] = [x1_2, y1_2, s1_2]
                self.status2 = s1_2
                
            elif state == 3:
                [self.x_cm, self.y_cm, self.status] = [x1_2, y1_2, s1_1]
                self.status2 = self.angle(np.array([x1_1,y1_1]), np.array([x1_2,y1_2]), np.array([640//2,0]), np.array([640//2,480]))
                
            elif state == 4:
                [self.x_cm, self.y_cm, self.status] = [x1_3, y1_3, s1_3]
                self.status2 = s1_3
                
            elif state == 5:
                [self.x_cm, self.y_cm, self.status] = [x1_1, y1_1, s1_1]
                self.status2 = self.angle(np.array([x1_1,y1_1]), np.array([x1_3,y1_3]), np.array([640//2,0]), np.array([640//2,480]))
                
            elif state == 6:
                [self.x_cm, self.y_cm, self.status] = [x1_2, y1_2, s1_2]
                self.status2 = self.angle(np.array([x1_2,y1_2]), np.array([x1_3,y1_3]), np.array([640//2,0]), np.array([640//2,480]))
                
                
            elif state == 7:
                [self.x_cm, self.y_cm, self.status] = [x1_2, y1_2, s1_1]
                self.status2 = self.angle3(np.array([x1_1,y1_1]), np.array([x1_2,y1_2]), np.array([x1_3,y1_3]))
                [A, B] = self.equation(np.array([y1_1, x1_1]), np.array([y1_3, x1_3]))
                y = (x1_2-B)/A
                if y < y1_2 :
                    self.status2 *= -1
                    
#==============================================================================
        
        else :
            self.model = "nothing"
        image = cv2.resize(image, (480, 360))
        cv2.putText(image, '%d'%self.status2, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('line_follower show', image)
       
        
        return self.model, self.x_cm, self.y_cm, self.status, self.status2
###############################################################################
# 起飛2
###############################################################################
    def takeoff_2(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 二值化
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, np.array([self.h_l_puting, self.s_l_puting, self.v_l_puting]), np.array([self.h_h_puting, self.s_h_puting, self.v_h_puting]))
        
        mask_black = cv2.inRange(hsv, np.array([self.h_l,self.s_l,self.v_l]), np.array([self.h_h,self.s_h,self.v_h]))
        
        result_blue = cv2.bitwise_and(img, img, mask=mask_blue)
        gray_blue = cv2.cvtColor(result_blue, cv2.COLOR_BGR2GRAY)
        median_blue = cv2.medianBlur(gray_blue,15)
        ret, binary_blue = cv2.threshold(median_blue, 0, 255, cv2.THRESH_BINARY)
        
        result_black = cv2.bitwise_and(img, img, mask=mask_black)
        result_black = cv2.subtract(result_black,result_blue)
        gray_black = cv2.cvtColor(result_black, cv2.COLOR_BGR2GRAY)
        median_black = cv2.medianBlur(gray_black,15)
        ret, binary_black = cv2.threshold(median_black, 0, 255, cv2.THRESH_BINARY)
        #######################################################################
        cnt_max_blue = self.contourmax(img, binary_blue, self.area_landing) 
        if len(cnt_max_blue) > 0:
            self.takeoff_found_2 = True  
            [d_x_blue, d_y_blue], point1_blue, point2_blue = self.matrix(img, cnt_max_blue)
        else :
            self.takeoff_found_2 = False
            d_x_blue, d_y_blue, point1_blue, point2_blue = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        #######################################################################
        cnt_max_black = self.contourmax(img, binary_black, 400) 
        if len(cnt_max_black) > 0:
            [d_x_black, d_y_black], point1_black, point2_black = self.matrix(img, cnt_max_black)
        else :
            d_x_black, d_y_black, point1_black, point2_black = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        #######################################################################
        d_x = (point1_black[0] + d_x_blue)//2
        d_y = (point1_black[1] + d_y_blue)//2
        point1 = point1_black
        point2 = np.array([d_x_blue, d_y_blue])
        
        # 計算X軸
        x_cm = -(d_y - h//2)*self.pixel_cm
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)
        
        # 計算Y軸
        y_cm = (d_x - w//2)*self.pixel_cm
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        
        # 顯示畫面
        img = cv2.resize(img, (480, 360))
        cv2.imshow('takeoff show 2', img)
        return x_cm, y_cm, status
###############################################################################
# 起飛
###############################################################################
    def takeoff(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 二值化
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, np.array([self.h_l_puting, self.s_l_puting, self.v_l_puting]), np.array([self.h_h_puting, self.s_h_puting, self.v_h_puting]))
        
        mask_black = cv2.inRange(hsv, np.array([self.h_l,self.s_l,self.v_l]), np.array([self.h_h,self.s_h,self.v_h]))
        
        result_blue = cv2.bitwise_and(img, img, mask=mask_blue)
        gray_blue = cv2.cvtColor(result_blue, cv2.COLOR_BGR2GRAY)
        median_blue = cv2.medianBlur(gray_blue,15)
        ret, binary_blue = cv2.threshold(median_blue, 0, 255, cv2.THRESH_BINARY)
        
        result_black = cv2.bitwise_and(img, img, mask=mask_black)
        result_black = cv2.subtract(result_black,result_blue)
        gray_black = cv2.cvtColor(result_black, cv2.COLOR_BGR2GRAY)
        median_black = cv2.medianBlur(gray_black,15)
        ret, binary_black = cv2.threshold(median_black, 0, 255, cv2.THRESH_BINARY)
        #######################################################################
        cnt_max_blue = self.contourmax(img, binary_blue, self.area_landing) 
        if len(cnt_max_blue) > 0:
            self.takeoff_found = True  
            [d_x_blue, d_y_blue], point1_blue, point2_blue = self.matrix(img, cnt_max_blue)
        else :
            self.takeoff_found = False
            d_x_blue, d_y_blue, point1_blue, point2_blue = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        #######################################################################
        cnt_max_black = self.contourmax(img, binary_black, 400) 
        if len(cnt_max_black) > 0:
            [d_x_black, d_y_black], point1_black, point2_black = self.matrix(img, cnt_max_black)
        else :
            d_x_black, d_y_black, point1_black, point2_black = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        #######################################################################
        d_x = (point1_black[0] + d_x_blue)//2
        d_y = (point1_black[1] + d_y_blue)//2
        point1 = point1_black
        point2 = np.array([d_x_blue, d_y_blue])
        
        # 計算X軸
        x_cm = -(d_y - h//2)*self.pixel_cm
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)
        
        # 計算Y軸
        y_cm = (d_x - w//2)*self.pixel_cm
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 顯示畫面
        img = cv2.resize(img, (480, 360))
        cv2.imshow('takeoff show', img)
        return x_cm, y_cm, status
###############################################################################
# 紅燈
###############################################################################
    def red_light(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 二值化
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_rad = cv2.inRange(hsv, np.array([self.h_l_red_light, self.s_l_red_light, self.v_l_red_light]), np.array([self.h_h_red_light, self.s_h_red_light, self.v_h_red_light]))
        result = cv2.bitwise_and(img, img, mask=mask_rad) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,15)
        # cv2.imshow('median', median)
        ret, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow('result', binary)
        #######################################################################
        # 找尋最大面積輪廓
        cnt_max = self.contourmax(img, binary, self.area_landing) 
                
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.red_light_found = True  
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
        else :
            self.red_light_found = False
            d_x, d_y, point1, point2 = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        
        # 計算X軸
        x_cm = -(d_y - h//2)*self.pixel_cm
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)
        
        # 計算Y軸
        y_cm = (d_x - w//2)*self.pixel_cm
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        
        if status > 0:
            status = 90 - status
            status = -status
        elif status < 0:   
            status += 90

            
            
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 顯示畫面
        img = cv2.resize(img, (480, 360))
        cv2.imshow('red_light show', img)
        return x_cm, y_cm, status
###############################################################################
# 投放
###############################################################################
    def putin(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 二值化
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, np.array([self.h_l_puting, self.s_l_puting, self.v_l_puting]), np.array([self.h_h_puting, self.s_h_puting, self.v_h_puting]))
        result = cv2.bitwise_and(img, img, mask=mask_blue) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,15)
        # cv2.imshow('median', median)
        ret, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow('result', binary)
        #######################################################################
        # 找尋最大面積輪廓
        cnt_max = self.contourmax(img, binary, self.area_landing) 
                
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.putin_found = True  
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
        else :
            self.putin_found = False
            d_x, d_y, point1, point2 = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        
        # 計算X軸
        x_cm = -(d_y - h//2)*self.pixel_cm
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)
        
        # 計算Y軸
        y_cm = (d_x - w//2)*self.pixel_cm
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 顯示畫面
        img = cv2.resize(img, (480, 360))
        cv2.imshow('putin show', img)
        return x_cm, y_cm, status
###############################################################################
# 降落
###############################################################################
    def landing(self,img):
          
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 二值化
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_rad = cv2.inRange(hsv, np.array([self.h_l_r, self.s_l_r, self.v_l_r]), np.array([self.h_h_r, self.s_h_r, self.v_h_r]))
        mask_green = cv2.inRange(hsv, np.array([self.h_l_g, self.s_l_g, self.v_l_g]), np.array([self.h_h_g, self.s_h_g, self.v_h_g]))
        result_rad = cv2.bitwise_and(img, img, mask=mask_rad) 
        
        result_green = cv2.bitwise_and(img, img, mask=mask_green) 
        result = cv2.bitwise_or(result_rad, result_green) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,15)
        # cv2.imshow('median', median)
        ret, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow('result', binary)
        #######################################################################
        # 找尋最大面積輪廓
        cnt_max = self.contourmax(img, binary, self.area_landing) 
                
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.landing_found = True  
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
        else :
            self.landing_found = False
            d_x, d_y, point1, point2 = w//2, h//2, np.array([w//2,0]), np.array([w//2,h])
        
        # 計算X軸
        x_cm = -(d_y - h//2)*self.pixel_cm
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)
        
        # 計算Y軸
        y_cm = (d_x - w//2)*self.pixel_cm
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # 顯示畫面
        img = cv2.resize(img, (480, 360))
        cv2.imshow('landing show', img)
        return x_cm, y_cm, status
###############################################################################
# 循線 1
###############################################################################
    def line_follower_1(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 高斯模糊--> 二值化--> 膨脹侵蝕
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask_black = cv2.inRange(hsv, np.array([self.h_l,self.s_l,self.v_l]), np.array([self.h_h,self.s_h,self.v_h]))
        result = cv2.bitwise_and(img, img, mask=mask_black) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,self.mid_k)
        # cv2.imshow('median', median)
        gaussian = cv2.GaussianBlur(median, (self.Gaus_k, self.Gaus_k), 0)
        # cv2.imshow('gaussian',gaussian)
        ret, binary = cv2.threshold(gaussian, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow('binary', binary2)
        kernel = np.ones((self.kernel,self.kernel), np.uint8)
        erode = cv2.erode(binary , kernel, iterations =  self.e1)
        dilate = cv2.dilate(erode, kernel, iterations = self.d1) 
        erode2 = cv2.erode(dilate , kernel, iterations = self.e2)
        dilate2 = cv2.dilate(erode2, kernel, iterations = self.d2) 
        # cv2.imshow('result', dilate2)
		#######################################################################
		# 找尋最大面積輪廓
        cnt_max = self.contourmax(img, dilate2, self.area_line_follower*(self.segmentation1+1/480)) 
        
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.line_follower_found_1 = True
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
            
        else :
            self.line_follower_found_1 = False
            d_x, d_y = w//2, h//2
            point1, point2 = np.array([w//2,0]), np.array([w//2,h])
            
        # 計算Y軸        
        y_cm = (d_x - w//2)*self.pixel_cm 
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算X軸  
        x_cm = -(d_y - h//2)
        x_cm += (self.segmentation1+1) //2
        x_cm += (480-self.segmentation2-self.segmentation1)//2
        x_cm *= self.pixel_cm 
        
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)

        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        
        
        cv2.line(img, (0,h), (w,h), (0, 0, 190), 1)
        cv2.line(img, (0,0), (w,0), (0, 0, 190), 1)
        # 顯示畫面
        # img = cv2.resize(img, (480, 360))
        # cv2.imshow('line_follower show_1', img)
        
        return x_cm, y_cm, status, img
###############################################################################
# 循線 2
###############################################################################
    def line_follower_2(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 高斯模糊--> 二值化--> 膨脹侵蝕
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask_black = cv2.inRange(hsv, np.array([self.h_l,self.s_l,self.v_l]), np.array([self.h_h,self.s_h,self.v_h]))
        result = cv2.bitwise_and(img, img, mask=mask_black) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,self.mid_k)
        # cv2.imshow('median', median)
        gaussian = cv2.GaussianBlur(median, (self.Gaus_k, self.Gaus_k), 0)
        # cv2.imshow('gaussian',gaussian)
        ret, binary = cv2.threshold(gaussian, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow('binary', binary2)
        kernel = np.ones((self.kernel,self.kernel), np.uint8)
        erode = cv2.erode(binary , kernel, iterations =  self.e1)
        dilate = cv2.dilate(erode, kernel, iterations = self.d1) 
        erode2 = cv2.erode(dilate , kernel, iterations = self.e2)
        dilate2 = cv2.dilate(erode2, kernel, iterations = self.d2) 
        # cv2.imshow('result', dilate2)
		#######################################################################
		# 找尋最大面積輪廓

        cnt_max = self.contourmax(img, dilate2, self.area_line_follower*(480-self.segmentation1-self.segmentation2/480)) 
        
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.line_follower_found_2 = True
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
            
        else :
            self.line_follower_found_2 = False
            d_x, d_y = w//2, h//2
            point1, point2 = np.array([w//2,0]), np.array([w//2,h])
            
        # 計算Y軸        
        y_cm = (d_x - w//2)*self.pixel_cm 
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算X軸  
        x_cm = -(d_y - h//2)*self.pixel_cm 
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)

        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.line(img, (0,h), (w,h), (0, 0, 190), 1)
        cv2.line(img, (0,0), (w,0), (0, 0, 190), 1)
        
        
        
        # cv2.imshow('line_follower show_2',img)
        # 顯示畫面
        # img = cv2.resize(img, (480, 360))
        
        
        
        
        return x_cm, y_cm, status, img
###############################################################################
# 循線 3
###############################################################################
    def line_follower_3(self,img):
        
        h, w = self.getshape(img)
        #######################################################################
        # 前處理 : 顏色遮罩--> 灰階--> 中值濾波--> 高斯模糊--> 二值化--> 膨脹侵蝕
        #######################################################################
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask_black = cv2.inRange(hsv, np.array([self.h_l,self.s_l,self.v_l]), np.array([self.h_h,self.s_h,self.v_h]))
        result = cv2.bitwise_and(img, img, mask=mask_black) 
        # cv2.imshow('mask', result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
        # cv2.imshow('gray', gray)
        median = cv2.medianBlur(gray,self.mid_k)
        # cv2.imshow('median', median)
        gaussian = cv2.GaussianBlur(median, (self.Gaus_k, self.Gaus_k), 0)
        # cv2.imshow('gaussian',gaussian)
        ret, binary = cv2.threshold(gaussian, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow('binary', binary2)
        kernel = np.ones((self.kernel,self.kernel), np.uint8)
        erode = cv2.erode(binary , kernel, iterations =  self.e1)
        dilate = cv2.dilate(erode, kernel, iterations = self.d1) 
        erode2 = cv2.erode(dilate , kernel, iterations = self.e2)
        dilate2 = cv2.dilate(erode2, kernel, iterations = self.d2) 
        # cv2.imshow('result', dilate2)
		#######################################################################
		# 找尋最大面積輪廓
        cnt_max = self.contourmax(img, dilate2, self.area_line_follower*(self.segmentation2-1/480)) 
        
        # 當面積超過閥值
        if len(cnt_max) > 0:
            self.line_follower_found_3 = True
            [d_x, d_y], point1, point2 = self.matrix(img, cnt_max)
            
        else :
            self.line_follower_found_3 = False
            d_x, d_y = w//2, h//2
            point1, point2 = np.array([w//2,0]), np.array([w//2,h])
            
        # 計算Y軸        
        y_cm = (d_x - w//2)*self.pixel_cm 
        cv2.putText(img, 'Y:%.2f cm'%y_cm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.line(img, (w//2,h//2), (d_x,h//2), (255, 0, 255), 2)
        
        # 計算X軸 
        x_cm = -(d_y - h//2)
        x_cm -= (self.segmentation2-1)//2
        x_cm -= (480-self.segmentation2-self.segmentation1)//2
        x_cm *= self.pixel_cm 
        
        cv2.putText(img, 'X:%.2f cm'%x_cm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.line(img, (w//2,h//2), (w//2,d_y), (0, 255, 255), 2)

        # 計算角度
        status = self.angle(point1, point2, np.array([w//2,0]), np.array([w//2,h]))
        cv2.putText(img, 'A:%d degree'%status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.line(img, (0,h), (w,h), (0, 0, 190), 1)
        cv2.line(img, (0,0), (w,0), (0, 0, 190), 1)
        # 顯示畫面
        # img = cv2.resize(img, (480, 360))
        # cv2.imshow('line_follower show_3', img)
        
        return x_cm, y_cm, status, img
###############################################################################
# 取得影像長寬
###############################################################################
    def getshape(self, img):
        
        [h,w,_] = img.shape
        # cv2.circle(img,(w//2, h//2), 1, (255, 255, 255), -1) # 中心點
        # cv2.line(img, (w//2,0), (w//2,h), (255, 255, 255), 1) # x軸
        # cv2.line(img, (0,h//2), (w,h//2), (255, 255, 255), 1) # y軸 
        
        return h,w
###############################################################################
# 計算三點夾角
###############################################################################
    def angle3(self, point_1, point_2, point_3):
        
        a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
        b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
        c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
        # A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
        # print(a)
        # print(b)
        # print(c)
        # print(math.acos(round(((b*b-a*a-c*c)/(-2*a*c)), 4)))
        B=math.degrees((math.acos(round(((b*b-a*a-c*c)/(-2*a*c)), 4))))
        # C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
       
        return B
###############################################################################
# 計算四點夾角
###############################################################################
    def angle(self, point1, point2, point3, point4):
        
        a = point2-point1
        b = point4-point3
        A = np.dot(a,b) 
        B = (((a[0]**2)+(a[1]**2))**(1/2)*((b[0]**2)+(b[1]**2))**(1/2))
        angle = (math.acos(A/B) * 180 /math.pi)
        if point1[0] < point2[0]:
            angle = -angle
            
        return angle
###############################################################################
# 取得影像最大面積輪廓
###############################################################################
    def contourmax(self, img, img_pre, area_max):
        
        cnt_max = []
        
        # 找尋輪廓並描繪
        contours, hierarchy = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(255,0,0),-1) 
        
        # 找出最大面積的輪廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_max:
                area_max = area
                cnt_max = cnt
                
        return cnt_max
###############################################################################
# 目標物中心
###############################################################################
    def matrix(self, img, cnt_max):
        
        # 凸包
        hull = cv2.convexHull(cnt_max)
        cv2.drawContours(img,[hull],0,(0,255,0),2)
        
        # 最小外接矩形
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        
        # 矩形中心點
        box_cen = (box[0]+box[1]+box[2]+box[3])//4 
        cv2.circle(img,tuple(box_cen), 4, (0, 0, 255), -1)

        # 矩陣中垂線
        if np.linalg.norm(box[0]-box[1]) < np.linalg.norm(box[0]-box[3]) :
            point1 = (box[0] + box[1])//2
            point2 = (box[2] + box[3])//2
        else:
            point1 = (box[1] + box[2])//2
            point2 = (box[0] + box[3])//2
        cv2.line(img, tuple(point1), tuple(point2), (255, 255, 0), 2)
        
        return box_cen, point1, point2
###############################################################################
# 高度轉化Pixel(cm)
###############################################################################
    def getpixelcm(self, real_high):
        
        model=[
        0.00269214,
        0.0102543
        ]
        fn=np.poly1d(model)
        pixel_cm =  fn(real_high)
        
        return pixel_cm
###############################################################################
# Trackbar使用(無義)
###############################################################################
    def nothing(self,n):
        
        pass
###############################################################################
# Trackbar設定
###############################################################################
    def seting(self):
        
        if self.ui :
            
            self.h_l = cv2.getTrackbarPos('Hue Min', 'line_follower')
            self.h_h = cv2.getTrackbarPos('Hue Max', 'line_follower')
            self.s_l = cv2.getTrackbarPos('Sat Min', 'line_follower')
            self.s_h = cv2.getTrackbarPos('Sat Max', 'line_follower')
            self.v_l = cv2.getTrackbarPos('Val Min', 'line_follower')
            self.v_h = cv2.getTrackbarPos('Val Max', 'line_follower')
            self.area_line_follower = cv2.getTrackbarPos('Area Min', 'line_follower')
            self.mid_k = self.kernel_size[cv2.getTrackbarPos('Median', 'line_follower')]
            self.Gaus_k = self.kernel_size[cv2.getTrackbarPos('Gaussian', 'line_follower')] 
            self.kernel = self.kernel_size[cv2.getTrackbarPos('Kernel', 'line_follower')]
            self.e1 = cv2.getTrackbarPos('Erode 1', 'line_follower')
            self.d1 = cv2.getTrackbarPos('Dilate 1', 'line_follower')
            self.e2 = cv2.getTrackbarPos('Erode 2', 'line_follower')
            self.d2 = cv2.getTrackbarPos('Dilate 2', 'line_follower')
            
            self.segmentation1 = cv2.getTrackbarPos('segmentation1', 'line_follower')
            self.segmentation2 = cv2.getTrackbarPos('segmentation2', 'line_follower')
            
            self.area_landing = cv2.getTrackbarPos('Area Min', 'landing')
            self.h_l_r = cv2.getTrackbarPos('Hue Min r', 'landing')
            self.h_h_r = cv2.getTrackbarPos('Hue Max r', 'landing')
            self.s_l_r = cv2.getTrackbarPos('Sat Min r', 'landing')
            self.s_h_r = cv2.getTrackbarPos('Sat Max r', 'landing')
            self.v_l_r = cv2.getTrackbarPos('Val Min r', 'landing')
            self.v_h_r = cv2.getTrackbarPos('Val Max r', 'landing')
            self.h_l_g = cv2.getTrackbarPos('Hue Min g', 'landing')
            self.h_h_g = cv2.getTrackbarPos('Hue Max g', 'landing')
            self.s_l_g = cv2.getTrackbarPos('Sat Min g', 'landing')
            self.s_h_g = cv2.getTrackbarPos('Sat Max g', 'landing')
            self.v_l_g = cv2.getTrackbarPos('Val Min g', 'landing')
            self.v_h_g = cv2.getTrackbarPos('Val Max g', 'landing')
            
            self.h_l_puting = cv2.getTrackbarPos('Hue Min puting', 'puting')
            self.h_h_puting = cv2.getTrackbarPos('Hue Max puting', 'puting')
            self.s_l_puting = cv2.getTrackbarPos('Sat Min puting', 'puting')
            self.s_h_puting = cv2.getTrackbarPos('Sat Max puting', 'puting')
            self.v_l_puting = cv2.getTrackbarPos('Val Min puting', 'puting')
            self.v_h_puting = cv2.getTrackbarPos('Val Max puting', 'puting')
            
            self.h_l_red_light = cv2.getTrackbarPos('Hue Min red_light', 'red_light')
            self.h_h_red_light = cv2.getTrackbarPos('Hue Max red_light', 'red_light')
            self.s_l_red_light = cv2.getTrackbarPos('Sat Min red_light', 'red_light')
            self.s_h_red_light = cv2.getTrackbarPos('Sat Max red_light', 'red_light')
            self.v_l_red_light = cv2.getTrackbarPos('Val Min red_light', 'red_light')
            self.v_h_red_light = cv2.getTrackbarPos('Val Max red_light', 'red_light')
            
        else : pass
###############################################################################
# 開始
###############################################################################
    def start(self):
        
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()
        time.sleep(1)
###############################################################################
# 壓縮尺寸
###############################################################################
    def img_resize(self,image):
        height, width = image.shape[0], image.shape[1]

        width_new = 1280
        height_new = 960
        if width / height >= width_new / height_new:
            img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
        else:
            img_new = cv2.resize(image, (int(width * height_new / height), height_new))
        img = img_new[240:720, 320:960,:]
        return img
###############################################################################
# 兩點求方程式
###############################################################################
    def equation(self, point1, point2):
    
        A = (point1[1]-point2[1])/(point1[0]-point2[0])
        B = -1 * ( (A*point1[0])-point1[1] )
    
        return A, B  
###############################################################################
# 結束
###############################################################################
    def stop(self):
        
        self.isstop = True
        cv2.destroyAllWindows()
        
        if self.ui:
            myDict = {"h_l": self.h_l,
                      "h_h": self.h_h,
                      "s_l": self.s_l,
                      "s_h": self.s_h,
                      "v_l": self.v_l,
                      "v_h": self.v_h,
                      
                      "area_line_follower": self.area_line_follower,
                      
                      "mid_k": self.mid_k,
                      "Gausk": self.Gaus_k,
                      "kernel": self.kernel,
                      "e1": self.e1,
                      "d1": self.d1,
                      "e2": self.e2,
                      "d2": self.d2,
                      
                      "h_l_r_landing": self.h_l_r,
                      "h_h_r_landing": self.h_h_r,
                      "s_l_r_landing": self.s_l_r,
                      "s_h_r_landing": self.s_h_r,
                      "v_l_r_landing": self.v_l_r,
                      "v_h_r_landing": self.v_h_r,
                      
                      "h_l_g_landing": self.h_l_g,
                      "h_h_g_landing": self.h_h_g,
                      "s_l_g_landing": self.s_l_g,
                      "s_h_g_landing": self.s_h_g,
                      "v_l_g_landing": self.v_l_g,
                      "v_h_g_landing": self.v_h_g,
                      
                      "area_landing": self.area_landing,
                      
                      "h_l_puting": self.h_l_puting,
                      "h_h_puting": self.h_h_puting,
                      "s_l_puting": self.s_l_puting,
                      "s_h_puting": self.s_h_puting,
                      "v_l_puting": self.v_l_puting,
                      "v_h_puting": self.v_h_puting,
                      
                      "h_l_red_light": self.h_l_red_light,
                      "h_h_red_light": self.h_h_red_light,
                      "s_l_red_light": self.s_l_red_light,
                      "s_h_red_light": self.s_h_red_light,
                      "v_l_red_light": self.v_l_red_light,
                      "v_h_red_light": self.v_h_red_light,
                      }
            with open("output.json", "w") as f:
                json.dump(myDict, f, indent = 4)
        print('Save!!!')
###############################################################################
# 獲取影像
###############################################################################         
    def read(self):
        
        return self.Frame.copy()
###############################################################################
# 多執行序
###############################################################################
    def queryframe(self):
        
        while (not self.isstop):
            _, self.Frame = self.capture.read()
        self.capture.release()
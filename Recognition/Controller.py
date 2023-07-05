import time
import numpy as np
import cv2


# Car = UITCar()

pre_t = time.time()
error_arr = np.zeros(5)



class Controller:
    def __init__(self, mask):
        self.mask = mask
        self.minLane = 0
        self.maxLane = 160
        self.__LANEWIGHT = 25
        self.width = 40

    def vision(self, height=25):
        arr_normal = []
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            # arr_normal = [frame.shape[1] * 1 // 3, frame.shape[1] * 2 // 3]
            return 0

        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)
        
        center = int((self.minLane + self.maxLane) / 2)
        
        #### Safe Mode ####
        width=self.maxLane-self.minLane
        self.width = width
        # print(width)

        # # if 0 <= self.minLane <= 10 and 80 <= self.maxLane <= 135 and width > 100:
        # #     center = self.maxLane - self.__LANEWIGHT//2
        # # elif 35 <= self.minLane <= 80 and self.maxLane >= 149 and width > 100:
        # #     center = self.minLane + self.__LANEWIGHT//2

        # #### Cua sớm ####
        # if (0 < width < self.__LANEWIGHT):
        #     # print('Cua som')
        #     if (center < int(self.mask.shape[1]/2)):
        #         center -= self.__LANEWIGHT - width
        #     else :
        #         center += self.__LANEWIGHT - width

        #### Error ####
        error = self.mask.shape[1]//2 - center
        # self.__pre_error = error

        return error
    
    def PID(self, error, kp=1, ki=0, kd=0.01):
        global pre_t
        # global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error*kp
        delta_t = time.time() - pre_t
        pre_t = time.time()
        D = (error-error_arr[1])/delta_t*kd
        angle = P + D
        
        
        print("--Angle First: {}".format(angle))
        
        if angle+5 >= 50:
            angle = 55
        elif angle+5 <= -45:
            angle = -45
        
        
        
        if angle < -2:
            angle = angle + abs(angle*2)
            angle = 100 + angle 
        elif angle > 2:
            angle = angle - abs(angle*2)
            angle = 70 + angle
        else:
            error = angle
            angle = 100 + error
            
        
        if angle > 155:
            angle = 160
        elif angle < 20:
            angle = 15
        
        
        # if angle < 0:
        #   angle = angle + 180
        # else:
        #   angle = angle - 180
       
       
            
        # if -5 < angle <= 5:
        #     angle = angle + 95
        
        # if angle < 0:
        #     angle = angle + abs(angle*2)
        #     angle = angle + 90
        # else:
        #     angle = angle - abs(angle*2)
        #     angle = angle + 90
        
        # if angle > 180:
        #     angle = 175
        # elif angle < 0:
        #     angle = 5
            
        # if 0 <= angle < 50:
        #     angle = 20
        # elif 160 <= angle <= 180:
        #     angle = 170
            
        # angle = angle + 90
        
        
        # if angle < 90 and angle >= 0:
        #     angle = angle + 90
        # elif angle > 90 and angle <= 180:
        #     angle = angle - 90
        # elif angle >180:
        #     angle = 180
        # elif angle <0:
        #     angle = 0
        
        return int(angle)
    
    @staticmethod
    def control_speed(error):
        return int(0.05*abs(error) + 48)
        # return int(48)
        # return int(0.15*abs(error) + 45)
    
    def __call__(self):
        error = self.vision()
        angle = self.PID(error)
        speed = self.control_speed(error)
        return angle, speed
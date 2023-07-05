import numpy as np
import os
import time
import logging

import cv2
import onnxruntime


from utils.display import set_display, show_fps

from Controller import *
from Motor import *

session_lane = onnxruntime.InferenceSession('../model/Pretrain_3.onnx', None, providers=['CUDAExecutionProvider'])
input_name_lane = session_lane.get_inputs()[0].name

Motor = Motor_DC()
Servo = Motor_Servo()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def remove_small_contours(image):
    try:
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove
    except Exception as e:
        logging.error(e)
        return image
        

def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đườngs
	image = image[200:, :, :]
	small_img = cv2.resize(image, ((image.shape[1]//4, image.shape[0]//4)))
	cv2.imshow('Resized img',small_img)
    
	small_img = small_img/255
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	prediction = prediction.astype(np.uint8)

	return prediction


def main():
    cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cam.isOpened():
        try:
            full_scrn = False
            fps = 0.0
            while True:
                tic = time.time()
                ret, img = cam.read()
                copy_image = np.copy(img)
                segmentation = road_lines(copy_image, session=session_lane, inputname=input_name_lane)
                segmentation = remove_small_contours(segmentation)

                controller = Controller(segmentation)
                angle, speed = controller()

                Servo.Rotate_angle(0,angle)
                Motor.setSpeed_pwm(speed)


                fps = 1.0 / (time.time() - tic)
                img = show_fps(img, fps)

                cv2.imshow("segmentation ",segmentation)
                cv2.imshow("Show main image ",img)

                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    Servo.Rotate_angle(0,90)
                    Motor.setSpeed_pwm(0)
                    time.sleep(2)
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
    else:
        print("Camera is error")

if __name__ == '__main__':
    main()

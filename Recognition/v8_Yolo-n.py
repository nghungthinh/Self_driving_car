
import cv2
import time
import logging
import numpy as np
import onnxruntime

from Recognition.utils.display import open_window, set_display, show_fps

import sys 
import imutils

from yoloDet import YoloTRT

session_sign = onnxruntime.InferenceSession('/home/jetson/Desktop/Project/model/sign_cnn.onnx', None, providers=['CPUExecutionProvider'])
input_name_sign = session_sign.get_inputs()[0].name

model = YoloTRT(library="./yolov7/build/libmyplugins.so",engine="./yolov7/build/yolov7-tiny.engine",conf=0.5,yolo_ver="v7")
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


def Classify(img,inputname,session):
    img = cv2.resize(img,(30, 30))
    img = img.astype('float32')/255
    img = img.reshape(1,30,30,3)
    
    prediction = session.run(None,{inputname:img})
    prediction = np.squeeze(prediction) 
    cll = np.argmax(prediction)

    return cll
def Get_name(predictions):
    if str(predictions) == "0":
        name_class = "Left"
    if str(predictions) == "1":
        name_class = "Right"
    if str(predictions) == "2":
        name_class = "Stop"
    if str(predictions) == "3":
        name_class = "Straight"
    return name_class
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
        

#   Hàm dự đoán làn đường dựa vào ảnh từ camera
def road_lines(image, session, inputname):
	# Crop ảnh lại, lấy phần ảnh có làn đườngs
	image = image[200:, :, :]
	small_img = cv2.resize(image, ((image.shape[1]//4, image.shape[0]//4)))
	cv2.imshow('image',small_img)
	small_img = small_img/255
	small_img = np.array(small_img, dtype=np.float32)
	small_img = small_img[None, :, :, :]
	prediction = session.run(None, {inputname: small_img})
	prediction = np.squeeze(prediction)
	prediction = np.where(prediction < 0.5, 0, 255)
	# prediction = prediction.reshape(small_img.shape[0], small_img.shape[1])
	prediction = prediction.astype(np.uint8)

	return prediction

def show_camera():

    window_title = "CSI Camera"
    avg_fps = []
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            #window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                tic = time.time()
                ret_val, frame = video_capture.read()
                frame = imutils.resize(frame, width=400)
                detections, t = model.Inference(frame)

                #print(detections)
                # Detection for yolov5
                if len(detections) != 0:
                    x_min = int(detections[0]["box"][0])
                    y_min = int(detections[0]["box"][1])
                    x_max = int(detections[0]["box"][2])
                    y_max = int(detections[0]["box"][3])
                    
                    cls_img = frame[y_min:y_max,x_min:x_max]
                    cll = Classify(cls_img,input_name_sign,session_sign)
                    name_class = Get_name(cll)
                    cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),2)
                    cv2.putText(frame,name_class, (x_min,y_min-10),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cll = "None"

                fps = 1.0 / (time.time() - tic)
                avg_fps.append(fps)
                img = show_fps(frame, fps)
                print("FPS: ", fps)
                cv2.imshow("Image", frame)
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    print('Min FPS: {} -- Max FPS: {}'.format(min(avg_fps),max(avg_fps)))
                    print("AVG_FPS: ",sum(avg_fps)/len(avg_fps))
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()

from Adafruit_PCA9685 import PCA9685
import time
import serial
import numpy as np

# global variable
servo_channel = 0  # Channel connected to the servo
servo_min = 40  # Minimum pulse width for the servo
servo_max = 600  # Maximum pulse width for the servo

class Motor_Servo():
    def __init__(self):
        # self.i2c = I2C
        self.pca = PCA9685(address=0x40,busnum=0)
        self.pca.set_pwm_freq(50)
    # @staticmethod
    def Rotate_angle(self, channel, angle):
        angle = max(0, min(angle, 180))
        pulse_width = int(servo_min + (servo_max - servo_min) * angle / 180.0)
        self.pca.set_pwm(channel, 0, pulse_width)
#-------------------------------------------------------
class Motor_DC():
    def __init__(self):
        self.__Port = "/dev/ttyTHS1"
        self.__Baudrate = 115200
        self.__Timeout = 1
        self.__serial_port = serial.Serial(self.__Port, self.__Baudrate, timeout = self.__Timeout)
    def setSpeed_pwm(self, CarSpeed):
        # speed = bytearray(f"{CarSpeed}", "ascii")
        speed = str(CarSpeed)
        self.__serial_port.write(speed.encode("utf-8"))
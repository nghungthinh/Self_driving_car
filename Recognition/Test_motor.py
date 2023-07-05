from Motor import *

motor = Motor_Servo()

while True:
    value = input('Nhap gia tri ')
    motor.Rotate_angle(channel = 0,angle = int(value))
    if value == '0':
        break
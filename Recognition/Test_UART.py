from Motor import *

uart = Motor_DC()
while True:
    value = input('Nhap gia tri ')
    uart.setSpeed_pwm(value)
    if value == '0':
        break
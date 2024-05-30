#!/usr/bin/env python
import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self, enable_pin=18, in1_pin=23, in2_pin=24):
        self.ENB = enable_pin
        self.INC = in1_pin
        self.IND = in2_pin
        GPIO.setwarnings(False)
        self.setup()
        self.ENB_pwm = self.pwm()

    def setup(self):
        '''初始化引脚'''
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ENB, GPIO.OUT)
        GPIO.setup(self.INC, GPIO.OUT)
        GPIO.setup(self.IND, GPIO.OUT)

    def pwm(self):
        '''初始化PWM（脉宽调制）'''
        pwm = GPIO.PWM(self.ENB, 500)
        pwm.start(0)
        return pwm

    def changespeed(self, speed):
        '''通过改变占空比改变马达转速'''
        self.ENB_pwm.ChangeDutyCycle(40)
        time.sleep(0.02)
        self.ENB_pwm.ChangeDutyCycle(speed)

    def clockwise(self):
        '''马达顺时针转的信号'''
        GPIO.output(self.INC, 1)
        GPIO.output(self.IND, 0)

    def counter_clockwise(self):
        '''马达逆时针转的信号'''
        GPIO.output(self.INC, 0)
        GPIO.output(self.IND, 1)

    def brake(self):
        '''马达制动的信号'''
        GPIO.output(self.INC, 0)
        GPIO.output(self.IND, 0)
        self.changespeed(100)

    def loop(self, cmd):
        '''通过输入的命令改变马达转动'''
        direction = cmd[0]
        if direction == "f":
            self.clockwise()
        if direction == "r":
            self.counter_clockwise()
        if direction == "b":
            self.brake()
            return
        speed = float(cmd[1:]) * 10
        self.changespeed(int(speed))

    def destroy(self):
        self.ENB_pwm.stop()
        GPIO.cleanup()

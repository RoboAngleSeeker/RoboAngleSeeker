#!/usr/bin/python3
# -*- coding: utf-8 -*-
# －－－－湖南创乐博智能科技有限公司－－－－
#  文件名：31_mpu6050.py
#  版本：V2.0
#  author: zhulin
# 说明：31_mpu6050.py
#####################################################

import smbus                    # 导入I2C的SMBus模块
from time import sleep          # 导入延时函数
import math

# 一些MPU6050寄存器及其地址
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

# MPU 6050 初始化工作
def makerobo_MPU_Init():
    # 写入抽样速率寄存器
    makerobo_bus.write_byte_data(makerobo_Device_Address, SMPLRT_DIV, 7)
    
    # 写入电源管理寄存器
    makerobo_bus.write_byte_data(makerobo_Device_Address, PWR_MGMT_1, 1)
    
    # 写入配置寄存器
    makerobo_bus.write_byte_data(makerobo_Device_Address, CONFIG, 0)
    
    # 写入陀螺配置寄存器
    makerobo_bus.write_byte_data(makerobo_Device_Address, GYRO_CONFIG, 24)
    
    # 写中断使能寄存器
    makerobo_bus.write_byte_data(makerobo_Device_Address, INT_ENABLE, 1)

# 读取MPU6050数据寄存器
def makerobo_read_raw_data(addr):
    # 加速度值和陀螺值为16位
        high = makerobo_bus.read_byte_data(makerobo_Device_Address, addr)
        low =  makerobo_bus.read_byte_data(makerobo_Device_Address, addr+1)
    
        # 连接更高和更低的值
        value = ((high << 8) | low)
        
        # 从mpu6050获取有符号值
        if(value > 32768):
                value = value - 65536
        return value


makerobo_bus = smbus.SMBus(1)      # 或bus = smbus.SMBus(0)用于较老的版本板
makerobo_Device_Address = 0x68   # MPU6050设备地址

makerobo_MPU_Init()             # 初始化MPU6050

# 打印出提示信息
print (" Makerobo Reading Data of Gyroscope and Accelerometer")

# 无限循环
# while True:
def get_angle():
    # 读取加速度计原始值
    acc_x = makerobo_read_raw_data(ACCEL_XOUT_H)
    acc_y = makerobo_read_raw_data(ACCEL_YOUT_H)
    acc_z = makerobo_read_raw_data(ACCEL_ZOUT_H)
    
    # 读陀螺仪原始值
    gyro_x = makerobo_read_raw_data(GYRO_XOUT_H)
    gyro_y = makerobo_read_raw_data(GYRO_YOUT_H)
    gyro_z = makerobo_read_raw_data(GYRO_ZOUT_H)
    
    # 全刻度范围+/- 250度/℃，根据灵敏度刻度系数
    Ax = acc_x/16384.0
    Ay = acc_y/16384.0
    Az = acc_z/16384.0
    
    Gx = gyro_x/131.0
    Gy = gyro_y/131.0
    Gz = gyro_z/131.0
    rotation_angle = math.atan2(Ay, Ax)
    return rotation_angle
    # print(rotation_angle*180/math.pi)
    # 打印出MPU相关信息
    # print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)     
    # sleep(1)     # 延时1s
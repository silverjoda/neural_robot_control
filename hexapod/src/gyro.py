#!/usr/bin/python
import smbus
import math
import time
import threading
 
class IMU():
    # Register
    power_mgmt_1 = 0x6b
    address = 0x68      

    def __init__(self):
        self.bus = smbus.SMBus(1) 
        self.bus.write_byte_data(IMU.address, IMU.power_mgmt_1, 0)

        self.current_time = time.time()
        self.imu_lock = threading.Lock()
        self.yaw = 0        

        self.loopthread = threading.Thread(target=self.loop, daemon=True)
        self.loopthread.start()


    def loop(self):
        self._update_yaw()
        time.sleep(0.005)


    def reset_yaw(self):
        with self.imu_lock:
            self.yaw = 0
 
    
    def get_yaw(self):
        with self.imu_lock:
            return self.yaw


    def _update_yaw(self):
        gyro_z_raw = self.read_word_2c(0x47)
        gyro_z_scaled = gyro_z_raw / 131
        new_time = time.time()
        dt = self.current_time - new_time
        self.current_time = new_time
        with self.imu_lock:
            self.yaw = self.yaw + dt * gyro_z_scaled
      

    def read_byte(self, reg):
        return self.bus.read_byte_data(IMU.address, reg)
     

    def read_word(self, reg):
        h = self.bus.read_byte_data(IMU.address, reg)
        l = self.bus.read_byte_data(IMU.address, reg+1)
        value = (h << 8) + l
        return value
     

    def read_word_2c(reg):
        val = self.read_word(reg)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val


    
         

 



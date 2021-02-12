from __future__ import division
import time
import numpy as np
import Adafruit_PCA9685

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

pwm_freq = 100
pwm.set_pwm_freq(pwm_freq)

def set_servo_cmd(channel, val):
    '''
    Command from [0,1]
    '''
    val_clip = np.clip(val, 0, 1)
    pulse_length = ((val_clip + 1) * 1000) / ((1000000. / pwm_freq) / 4096.)
    pwm.set_pwm(channel, 0, int(pulse_length))

servo_id = 0
set_servo_cmd(servo_id, 0)
time.sleep(7)
print("Starting test: ")
set_servo_cmd(servo_id, 1)
time.sleep(0.5)
set_servo_cmd(servo_id, 0)
time.sleep(0.5)

set_servo_cmd(servo_id, 1)
time.sleep(0.5)
set_servo_cmd(servo_id, 0.5)
time.sleep(0.3)

set_servo_cmd(servo_id, 1)
time.sleep(0.3)
set_servo_cmd(servo_id, 0.5)
time.sleep(0.3)

set_servo_cmd(servo_id, 0)

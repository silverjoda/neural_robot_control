# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time
import numpy as np

# Import the PCA9685 module.
import Adafruit_PCA9685

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

motor_id = 1 # 0: motor, 1: servo
pwm_freq = 100
scaler = 1

pwm.set_pwm_freq(pwm_freq)

def set_servo_cmd(channel, val):
    '''
    Command from [0,1]
    '''
    val_clip = np.clip(val, 0, 1)
    pulse_length = ((val_clip + 1) * 1000) / ((1000000. / pwm_freq) / 4096.)
    pwm.set_pwm(channel, 0, int(pulse_length))
    print(int(pulse_length))
   
# Set servo to zero
#set_servo_cmd(motor_id, 0)
#time.sleep(2)

print(f'Moving motor on channel {motor_id}, press Ctrl-C to quit...')
while True:
    set_servo_cmd(motor_id, 0.55)
    time.sleep(1)
    set_servo_cmd(motor_id, 0)
    time.sleep(1)

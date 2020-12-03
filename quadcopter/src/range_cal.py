# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time
import numpy as np

# Import the PCA9685 module.
import Adafruit_PCA9685

# Uncomment to enable debug output.
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Alternatively specify a different address and/or bus:
#pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)

pwm_freq = 100 
pwm.set_pwm_freq(pwm_freq)

def set_servo_cmd(channel, val):
    '''
    Command from [0,1]
    '''
    val_clip = np.clip(val, 0, 1)
    pulse_length = ((val_clip + 1) * 1000) / ((1000000. / pwm_freq) / 4096.)
    pwm.set_pwm(channel, 0, int(pulse_length))
    
print("Setting high")
for i in range(4):
    set_servo_cmd(i, 1)

print("Sleeping 4 secs")
time.sleep(4)

print("Setting low")
for i in range(4):
    set_servo_cmd(i, 0)

print("Sleeping 6 secs")
time.sleep(6)

print("Done")



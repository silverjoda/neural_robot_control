# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import time
import numpy as np
import RPi.GPIO as GPIO
import Adafruit_PCA9685


# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

motor_id = 1 # 0: motor, 1: servo
pwm_freq = 100
scaler = 1
gpio_input_id = 9
gpio_output_id = 11

pwm.set_pwm_freq(pwm_freq)

GPIO.setmode(GPIO.BCM)
GPIO.setup(gpio_input_id, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(gpio_output_id, GPIO.OUT)

def set_servo_cmd(channel, val):
    '''
    Command from [0,1]
    '''
    val_clip = np.clip(val, 0, 1)
    pulse_length = ((val_clip + 1) * 1000) / ((1000000. / pwm_freq) / 4096.)
    pwm.set_pwm(channel, 0, int(pulse_length))
    #print(int(pulse_length))
   
# Set servo to zero and GPIO to low and wait
set_servo_cmd(motor_id, 0)
GPIO.output(gpio_output_id, GPIO.LOW)

#input("Press Enter to continue...")
#time.sleep(1)

print("Waiting for hardware input")
while GPIO.input(gpio_input_id) > 0: pass

# Write output and then immediately servo
#time.sleep(0.000001)
#GPIO.output(gpio_output_id, GPIO.HIGH)
set_servo_cmd(motor_id, 1.)

print("Output written. ")
input("Press Enter to continue...")

GPIO.output(gpio_output_id, GPIO.LOW)
set_servo_cmd(motor_id, 0.)

GPIO.cleanup()


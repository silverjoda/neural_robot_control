import RPi.GPIO as GPIO
import time

inputs = [11,17,27,10,22,9]

GPIO.setmode(GPIO.BCM)
for ipt in inputs:
    GPIO.setup(ipt, GPIO.IN)

while True:
    switch_status = [GPIO.input(ipt) for ipt in inputs]
    print('Switch status = ', switch_status)

GPIO.cleanup()

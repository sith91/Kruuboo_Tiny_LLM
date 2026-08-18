# hardware/real_hardware.py
"""Real hardware interface for Raspberry Pi / Arduino"""

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import RPi.GPIO as GPIO
import smbus2
import serial

class RealHardwareBackend:
    """Actual hardware implementation for Raspberry Pi"""
    
    def __init__(self):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Motor pins (L298N motor driver)
        self.MOTOR_LEFT_PWM = 12
        self.MOTOR_RIGHT_PWM = 13
        self.MOTOR_LEFT_DIR = 23
        self.MOTOR_RIGHT_DIR = 24
        
        GPIO.setup(self.MOTOR_LEFT_PWM, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_PWM, GPIO.OUT)
        GPIO.setup(self.MOTOR_LEFT_DIR, GPIO.OUT)
        GPIO.setup(self.MOTOR_RIGHT_DIR, GPIO.OUT)
        
        self.pwm_left = GPIO.PWM(self.MOTOR_LEFT_PWM, 1000)
        self.pwm_right = GPIO.PWM(self.MOTOR_RIGHT_PWM, 1000)
        self.pwm_left.start(0)
        self.pwm_right.start(0)
        
        # Temperature/humidity sensor (DHT22)
        import Adafruit_DHT
        self.dht_sensor = Adafruit_DHT.DHT22
        self.dht_pin = 4
        
        # Distance sensor (HC-SR04)
        self.TRIG = 17
        self.ECHO = 27
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        
        # Light sensor (photoresistor via ADC)
        i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(i2c)
        self.light_sensor = AnalogIn(self.ads, ADS.P0)
        
        # Servo
        self.servo_pin = 18
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.servo = GPIO.PWM(self.servo_pin, 50)
        self.servo.start(0)
    
    def read_temperature(self) -> float:
        """Read temperature from DHT22"""
        humidity, temperature = Adafruit_DHT.read_retry(self.dht_sensor, self.dht_pin)
        return temperature if temperature else 0
    
    def read_humidity(self) -> float:
        """Read humidity from DHT22"""
        humidity, temperature = Adafruit_DHT.read_retry(self.dht_sensor, self.dht_pin)
        return humidity if humidity else 0
    
    def read_distance(self) -> float:
        """Read distance from HC-SR04"""
        GPIO.output(self.TRIG, False)
        time.sleep(0.2)
        
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)
        
        pulse_start = time.time()
        pulse_end = time.time()
        
        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
        
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)
    
    def read_light(self) -> float:
        """Read light level"""
        return self.light_sensor.voltage
    
    def set_motor(self, motor: str, speed: int, direction: str = "forward"):
        """Control DC motor"""
        speed = max(0, min(100, speed))
        
        if motor == "left":
            GPIO.output(self.MOTOR_LEFT_DIR, GPIO.HIGH if direction == "forward" else GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(speed)
        elif motor == "right":
            GPIO.output(self.MOTOR_RIGHT_DIR, GPIO.HIGH if direction == "forward" else GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(speed)
    
    def set_servo(self, angle: int):
        """Control servo motor"""
        angle = max(0, min(180, angle))
        duty = angle / 18 + 2
        self.servo.ChangeDutyCycle(duty)
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.pwm_left.stop()
        self.pwm_right.stop()
        self.servo.stop()
        GPIO.cleanup()
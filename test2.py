from djitellopy import Tello, TelloException
import time
from colorama import Fore, Back, Style
import cv2

class ControlDrone:
    def __init__(self):
        self.tello = Tello()
        # self.info_drone()
        self.start()
        self.play()


    def start(self):
        try:
            print('1. Connection test:')
            self.tello.connect()
            # self.tello.streamon()
            print('Connection successful')
            self.tello.streamon()
        except Exception as e:
            print(f'Error connecting to the drone: {e}')
            raise

        try:
            self.tello.takeoff()
            print('Takeoff successful')
        except Exception as e:
            print(f'Error during takeoff: {e}')
            raise

    def play(self):
        # self.tello.move_left(20)
        self.tello.move_right(20)
        # self.tello.move_back(20)
        # self.tello.move_forward(20)
        self.tello.land()
        self.tello.end()


    def land(self):
        self.tello.land()
        self.tello.end()




if __name__ == "__main__":
    control = ControlDrone()


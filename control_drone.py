import cv2
import time
from colorama import Fore
from djitellopy import Tello

class ControlDrone:
    def __init__(self):
        self.tello = Tello()
        self._is_landing = False

        # RC control velocities
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0
        # self.tello.set_speed(100)


    def start(self):
        try:
            print('1. Connection test:')
            self.tello.connect()
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

    def get_frame(self):
        try:
            frame = self.tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as e:
            print(f'frame NOT found: {e}')
            raise
        


    def follow_order(self, order):
        if not self._is_landing:
            if order == 0: #UP
                print(Fore.LIGHTMAGENTA_EX ,"DO THE ACTION UP")
                self.up_down_velocity = 20

            elif order == 1:   #Down
                print(Fore.LIGHTBLUE_EX ," DO THE ACTION Donw")
                self.up_down_velocity = -20

            elif order == 2:  #Forward
                print(Fore.LIGHTCYAN_EX ," DO THE ACTION Forward")
                self.forw_back_velocity = 20

            elif order == 3:  #Back
                print(Fore.LIGHTWHITE_EX ," DO THE ACTION Back")
                self.forw_back_velocity = -20

            elif order == 4:  #Right
                print(Fore.LIGHTYELLOW_EX ," DO THE ACTION Right")
                self.left_right_velocity = -15

            elif order == 5:  #Left
                print(Fore.LIGHTGREEN_EX ," DO THE ACTION Left")
                self.left_right_velocity = 15

            elif order == 6:  #Rotate 360 degree
                print(Fore.LIGHTRED_EX ," DO THE ACTION Rotate 360 degree")
                self.tello.rotate_counter_clockwise(360)


            elif order == 7:  #Land
                print(Fore.LIGHTBLACK_EX ," DO THE ACTION Land")
                self._is_landing = True
                self.forw_back_velocity = self.up_down_velocity = \
                self.left_right_velocity = self.yaw_velocity = 0
                self.tello.land()
                self.tello.end()

            elif order == 8:  #Take Picture
                print(Fore.MAGENTA ," DO THE ACTION Take a Picture")
                t_end = time.time() + 1
                while time.time() < t_end:
                    frame_read = self.tello.get_frame_read()
                    frame_rgb = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("picture.png", frame_rgb)

            elif order == 9:    #None
                print(Fore.BLUE ," DO NOTHING")
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
            self.tello.send_rc_control(self.left_right_velocity, self.forw_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)
            
    def land(self):
        self.tello.land()
        self.tello.streamoff()
        self.tello.end()
        cv2.destroyAllWindows()


from djitellopy import Tello, TelloException
import time
from colorama import Fore, Back, Style

import cv2

class ControlDrone:
    def __init__(self):
        self.tello = Tello()
        # self.info_drone()
        self._is_landing = False

        # RC control velocities
        self.forw_back_velocity = 0
        self.up_down_velocity = 0
        self.left_right_velocity = 0
        self.yaw_velocity = 0
        self.tello.set_speed(80)


        # self.start()



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

    def get_frame(self):
        try:
            # print('1. Frame test:')
            frame = self.tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print('frame found successful')
            # cv2.imshow("Drone Camera", frame)
            return frame
        except Exception as e:
            print(f'frame NOT found: {e}')
            raise
        


    def follow_order(self, order):
        if not self._is_landing:
            if order == 0: #UP
                print(Fore.LIGHTMAGENTA_EX ,"DO THE ACTION UP")
                self.up_down_velocity = 15
                # self.tello.move_up(20)

            elif order == 1:   #Down
                print(Fore.LIGHTBLUE_EX ," DO THE ACTION Donw")
                self.up_down_velocity = -15
                # self.tello.move_down(20)

            elif order == 2:  #Forward
                print(Fore.LIGHTCYAN_EX ," DO THE ACTION Forward")
                self.forw_back_velocity = 20
                # self.tello.move_forward(20)

            elif order == 3:  #Back
                print(Fore.LIGHTWHITE_EX ," DO THE ACTION Back")
                self.forw_back_velocity = -20
                # self.tello.move_back(20)

            elif order == 4:  #Right
                print(Fore.LIGHTYELLOW_EX ," DO THE ACTION Right")
                self.left_right_velocity = -10
                # self.tello.move_right(20)

            elif order == 5:  #Left
                print(Fore.LIGHTGREEN_EX ," DO THE ACTION Left")
                self.left_right_velocity = 10
                # self.tello.move_left(20)

            elif order == 6:  #Rotate 360 degree
                print(Fore.LIGHTRED_EX ," DO THE ACTION Rotate 360 degree")
                self.tello.rotate_counter_clockwise(10)

            elif order == 7:  #Land
                print(Fore.LIGHTBLACK_EX ," DO THE ACTION Land")
                self._is_landing = True
                self.forw_back_velocity = self.up_down_velocity = \
                self.left_right_velocity = self.yaw_velocity = 0
                self.tello.land()

                # self.tello.land()
                # self.tello.streamoff()
                # self.tello.end()
                # cv2.destroyAllWindows()

            elif order == 8:  #Take Picture
                print(Fore.MAGENTA ," DO THE ACTION Take a Picture")
                t_end = time.time() + 5
                while time.time() < t_end:
                    frame_read = self.tello.get_frame_read()
                    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("picture.png", frame_rgb.frame)

            elif order == 9:    #None
                print(Fore.BLUE ," DO NOTHING")
                self.forw_back_velocity = self.up_down_velocity = \
                    self.left_right_velocity = self.yaw_velocity = 0
                # self.tello.send_rc_control(0, 0, 0, 0)

            # print('hiiii', self.tello.get_speed)
            self.tello.send_rc_control(self.left_right_velocity, self.forw_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)

    def info_drone(self):
        # print('Battery: ', self.tello.get_battery(), '%')
        # print('Temperature', self.tello.get_temperature())
        # print('Flight Time', self.tello.get_flight_time(), 'seconds')

        try:
            battery = self.tello.get_battery()
            print('Battery: ', battery, '%')
        except TelloException as e:
            print("Error retrieving battery status: ", e)

        try:
            temperature = self.tello.get_temperature()
            print('Temperature: ', temperature, '°C')
        except TelloException as e:
            print("Error retrieving temperature: ", e)

        try:
            flight_time = self.tello.get_flight_time()
            print('Flight Time: ', flight_time, 'seconds')
        except TelloException as e:
            print("Error retrieving flight time: ", e)


    def land(self):
        self.tello.land()
        self.tello.streamoff()
        self.tello.end()
        cv2.destroyAllWindows()


        

        



        
        
from djitellopy import Tello
import time
from colorama import Fore, Back, Style

class ControlDrone:
    def __init__(self):
        self.tello = Tello()
        self.info_drone()

    def start(self):
        try:
            print('1. Connection test:')
            self.tello.connect()
            print('Connection successful')
        except Exception as e:
            print(f'Error connecting to the drone: {e}')
            raise

        try:
            self.tello.takeoff()
            print('Takeoff successful')
        except Exception as e:
            print(f'Error during takeoff: {e}')
            raise

    def follow_order(self, order):
        if order == 0: #Forward
            print(Fore.LIGHTMAGENTA_EX ,"DO THE ACTION Forward")

        elif order == 1:   #Stop
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Stop")

        elif order == 2:  #Up
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Up")

        elif order == 3:  #Land
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Land")

        elif order == 4:  #Down
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Down")

        elif order == 5:  #Back
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Back")

        elif order == 6:  #Left
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Left")

        elif order == 7:  #Right
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Right")

        elif order == 8:  #Piece
            print(Fore.LIGHTBLUE_EX ," DO THE ACTION Take a Picture")

        elif order == 9:    #None
            print(Fore.LIGHTBLUE_EX ," DO NOTHING")

        # if order == 0: #Forward
        #     self.tello.move_forward(10)
        # if order == 0: #Forward
        #     self.tello.move_forward(10)
        # elif order == 1:   #Back
        #     self.tello.move_back(10)
        # elif order == 2:  
        #     self.tello.move_right(10)
        # elif order == 3:  
        #     self.tello.move_left(10)
        # elif order == 4:  
        #     self.tello.move_up(10)
        # elif order == 5:  
        #     self.tello.move_down(10)
        # elif order == 6:  
        #     self.tello.rotate_clockwise(10)
        # elif order == 7:  
        #     self.tello.rotate_counter_clockwise(10)
        # elif order == 8:  
        #     self.tello.land()
        #     self.tello.end()
        # elif order == 9:     # Take a Picture
        #     t_end = time.time() + 5
        #     while time.time() < t_end:
        #         self.tello.get_frame_read()


    

    def info_drone(self):
        print('Battery: ', self.tello.get_battery, '%')
        print('Temperature', self.tello.get_temperature)
        print('Flight Time', self.tello.get_flight_time)


    def land(self):
        self.tello.land()
        self.tello.end()


        

        



        
        
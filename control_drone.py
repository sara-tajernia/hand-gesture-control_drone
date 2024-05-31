from djitellopy import Tello
import time


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
            print('Forward')

        elif order == 1:   #Stop
            print('Stop')

        elif order == 2:  #Up
            print('Up')

        elif order == 3:  #Land
            print('Land')

        elif order == 4:  #Down
            print('Down')

        elif order == 5:  #Back
            print('Back')

        elif order == 6:  #Left
            print('Left')

        elif order == 7:  #Right
            print('Right')

        elif order == 8:  #Piece
            print('Take a Picture')

        elif order == 9:    # Take a Picture
            print('NONE')

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


        

        



        
        
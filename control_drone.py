from djitellopy import Tello
import time


class ControlDrone:

    def __init__(self, order):
        self.order = order
        self.tello = Tello()
        self.info_drone()
        self.follow_order()

    def start(self):
        print('1. Connection test:')
        self.tello.connect()
        self.tello.takeoff()

    def follow_order(self):

        match self.order:
            case 0:    #Forward
                self.tello.move_forward(10)
            case 1:    #Back
                self.tello.move_back(10)
            case 2:     
                self.tello.move_right(10)
            case 3:     
                self.tello.move_left(10)
            case 4:     
                self.tello.move_up(10)
            case 5:     
                self.tello.move_down(10)
            case 6:     
                self.tello.rotate_clockwise(10)
            case 7:     
                self.tello.land()
                self.tello.end()
            case 8:     # Take a Picture
                t_end = time.time() + 5
                while time.time() < t_end:
                    self.tello.get_frame_read()
            case 9:
                self.tello.streamon()


    

    def info_drone(self):
        print('Battery: ', self.tello.get_battery, '%')
        print('Temperature', self.tello.get_temperature)
        print('Flight Time', self.tello.get_flight_time)


        

        



        
        
from djitellopy import Tello

if __name__ == '__main__':

    print('1. Connection test:')
    tello = Tello()
    tello.connect()

    print('2. Video stream test:')
    # tello.streamon()
    # print('\n')

    # tello.takeoff()
    # tello.move_up(10)
    # tello.move_left(10)
    # tello.rotate_counter_clockwise(90)
    # tello.move_forward(100)

    print(tello.get_battery())

    tello.land()

    tello.end()
import itertools
import numpy
import time

import pypot.dynamixel

AMP = 30
FREQ = 0.5

if __name__ == '__main__':
    ports = pypot.dynamixel.get_available_ports()
    print('available ports:', ports)

    if not ports:
        raise IOError('No port available.')

    port = ports[0]
    print('Using the first on the list', port)

    dxl_io = pypot.dynamixel.DxlIO(port, use_sync_read=False, timeout=0.05, convert=True)
    print('Connected!')

    ids = range(1,19)

    scanned_ids = dxl_io.scan(ids)
    print("Scanned ids: {}".format(scanned_ids))
    assert len(scanned_ids) == len(ids)

    dxl_io.enable_torque(ids)

    speed = dict(zip(ids, itertools.repeat(100)))
    torque = dict(zip(ids, itertools.repeat(100)))
    dxl_io.set_moving_speed(speed)
    dxl_io.set_max_torque(torque)
    

    pos = dict(zip(ids, itertools.repeat(0)))
    
    for i in range(100):

        t1 = time.time()
        dxl_io.set_goal_position(pos)
        t2 = time.time()

        print("Write speed: {}".format(t2-t1))
    

        t1 = time.time()
        present_pos = dxl_io.get_present_position(ids)
        print(present_pos)
        t2 = time.time()

        print("Read speed: {}".format(t2-t1))

        time.sleep(1)
    

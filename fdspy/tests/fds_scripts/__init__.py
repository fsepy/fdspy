from os.path import join, dirname


with open(join(dirname(__file__), 'general-benchmark_1.fds'), 'r') as f:
    general_benchmark_1 = f.read()


with open(join(dirname(__file__), 'general-error.fds'), 'r') as f:
    general_error = f.read()


with open(join(dirname(__file__), 'general-residential_corridor.fds'), 'r') as f:
    general_residential_corridor = f.read()


with open(join(dirname(__file__), 'general-room_fire.fds'), 'r') as f:
    general_room_fire = f.read()


with open(join(dirname(__file__), 'travelling_fire-1cw.fds'), 'r') as f:
    travelling_fire_1cw = f.read()


with open(join(dirname(__file__), 'travelling_fire-line-1_ignition_origin.fds'), 'r') as f:
    travelling_fire_line_1_ignition_origin = f.read()


with open(join(dirname(__file__), 'travelling_fire-line-2_ignition_origin.fds'), 'r') as f:
    travelling_fire_line_2_ignition_origin = f.read()

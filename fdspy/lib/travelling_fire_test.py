from travelling_fire import *


def test_travelling_fire_line_1_ignition():
    from fdspy.tests.fds_scripts import travelling_fire_line_1_ignition_origin

    test = FDSTravellingFireMaker(fds_raw=travelling_fire_line_1_ignition_origin)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire'],
        ignition_origin=[(0, 0.5, 0)],
        ignition_delay_time=[0],
        spread_speed=[0.2],
        burning_time=[10]
    )

    print(res)


def test_travelling_fire_line_2_ignition():
    from fdspy.tests.fds_scripts import travelling_fire_line_2_ignition_origin

    test = FDSTravellingFireMaker(fds_raw=travelling_fire_line_2_ignition_origin)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire_a', 'travelling_fire_b'],
        ignition_origin=[(0, 0.5, 0), (5, 0.5, 0)],
        ignition_delay_time=[0, 0],
        spread_speed=[0.2, 0.2],
        burning_time=[10, 10]
    )

    print(res)


def test_travelling_fire_1cw():
    from fdspy.tests.fds_scripts import travelling_fire_1cw

    test = FDSTravellingFireMaker(fds_raw=travelling_fire_1cw)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire_a', 'travelling_fire_b', 'travelling_fire_c'],
        ignition_origin=[(3.3, 46.7, 3.82), (24.3, 28.7, 3.82), (36.3, 27.7, 3.82)],
        ignition_delay_time=[0, 27.5, 27.5 + 12.0],
        spread_speed=[1, 1, 1],
        burning_time=[5, 5, 5]
    )

    print(res)


if __name__ == '__main__':
    test_travelling_fire_line_1_ignition()
    test_travelling_fire_line_2_ignition()
    test_travelling_fire_1cw()

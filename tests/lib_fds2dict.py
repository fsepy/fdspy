# -*- coding: utf-8 -*-


def test_all_fds_input_parameters_in_a_list():
    from fdspy.lib.fds2dict import all_fds_input_parameters_in_a_list
    assert len(all_fds_input_parameters_in_a_list()) == 652


def test_fds_groups_in_a_list():
    from fdspy.lib.fds2dict import all_fds_groups_in_a_list

    assert len(all_fds_groups_in_a_list()) == 36


def test_fds2dict_parameterise_single_fds_command():
    from fdspy.lib.fds2dict import fds2dict_parameterise_single_fds_command as ff

    def fff(line_):
        line_ = ff(line_)
        if isinstance(line_, list):
            return len(line_)
        elif line_ is None:
            return None

    line = r"&HEAD CHID='moe1'/"
    assert fff(line) == 3

    line = r"&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"
    assert fff(line) == 5

    line = r"&MESH ID='stair upper02', IJK=7,15,82, XB=4.2,4.9,-22.0,-20.5,11.1,19.3, MPI_PROCESS=0/"
    assert fff(line) == 9

    line = r"""
    &PART ID='Tracer',
          MASSLESS=.TRUE.,
          MONODISPERSE=.TRUE.,
          AGE=60.0/
    """
    assert fff(line) == 9

    line = r"&CTRL ID='invert', FUNCTION_TYPE='ALL', LATCH=.FALSE., INITIAL_STATE=.TRUE., INPUT_ID='ventilation'/"
    assert fff(line) == 11

    line = r"&HOLE ID='door - stair_bottom', XB=3.0,3.4,-23.1,-22.3,4.9,6.9/ "
    assert fff(line) == 5

    line = r"&SLCF QUANTITY='TEMPERATURE', VECTOR=.TRUE., PBX=3.4/"
    assert fff(line) == 7

    line = r"&TAIL /"
    assert fff(line) == 1

    line = r"""
    &SURF ID='LINING CONCRETE',
          COLOR='GRAY 80',
          BACKING='VOID',
          MATL_ID(1,1)='CONCRETE',
          MATL_MASS_FRACTION(1,1)=1.0,
          THICKNESS(1)=0.2/
    """
    assert fff(line) == 13

    line = r"""&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"""
    assert ff(line)[3] == 'RESTRICT_TIME_STEP'

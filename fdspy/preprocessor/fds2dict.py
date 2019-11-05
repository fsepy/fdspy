# -*- coding: utf-8 -*-

"""

"""

import re
import copy
import warnings


class FDS2Dict:
    pass


def all_fds_groups_in_a_list(fds_manual_latex: str = None):

    # Parse input, i.e. the manual latex source code
    # ==============================================
    if fds_manual_latex is None:
        from fdspy.preprocessor import FDS_MANUAL_TABLE_GROUP_NAMELIST as _
        out = _
    else:
        out = fds_manual_latex

    # Analyse the source code, extract FDS input parameters
    # =====================================================

    # replace all escaped characters
    out = out.replace("\\", "")
    # remove all commented-out lines
    out = re.sub(r"%[\s\S.]*?[\r|\n]", "", out)
    # remove multiple \n or \r, step 1 - split string
    out = re.split(r"[\r|\n]", out)
    # remove multiple \n or \r, step 2 - remove empty lines
    out = list(filter(None, out))
    # remove multiple \n or \r, step 3 - restore to a single string
    out = '\n'.join(out)
    # find all possible FDS input parameters
    out = re.findall(r"\n{ct\s([\w]*)[(\}]", out)
    # filter out duplicated and sort all the items
    out = sorted(list(set(out)))

    return out


def test_fds_groups_in_a_list():
    assert len(all_fds_groups_in_a_list()) == 36


def all_fds_input_parameters_in_a_list(fds_manual_latex: str = None):
    """Get an exhausted list of input parameters for all groups in Fire Dynamics Simulator.

    :param fds_manual_latex: text string in latex source code obtained from FDS manual source codes.
    :return: a list of all input parameters extracted from the supplied FDS manual latex source code.
    """

    # Parse input, i.e. the manual latex source code
    # ==============================================
    if fds_manual_latex is None:
        from fdspy.preprocessor import FDS_MANUAL_CHAPTER_LIST_OF_INPUT_PARAMETERS as _
        fds_manual_latex = _
    else:
        fds_manual_latex = fds_manual_latex

    # remove latex formatter
    fds_manual_latex = re.sub(r"\\footnotesize *[\n\r]*?", " ", fds_manual_latex)

    # Analyse the source code, extract FDS input parameters
    # =====================================================

    # replace all escaped characters
    fds_manual_latex = fds_manual_latex.replace("\\", "")
    # remove all commented-fds_manual_latex lines
    fds_manual_latex = re.sub(r"%[\s\S.]*?[\r\n]", "", fds_manual_latex)
    # remove multiple \n or \r, step 1 - split string
    fds_manual_latex = re.split(r"[\r\n]", fds_manual_latex)
    # remove multiple \n or \r, step 2 - remove empty lines
    fds_manual_latex = list(filter(None, fds_manual_latex))
    # remove multiple \n or \r, step 3 - restore to a single string
    fds_manual_latex = '\n'.join(fds_manual_latex)
    # find all possible FDS input parameters
    fds_manual_latex = re.findall(r"\n{ct\s([\w]+)[(\} *,]", fds_manual_latex) + ['PBY', 'PBZ', 'FYI']
    # filter fds_manual_latex duplicated and sort all the items
    fds_manual_latex = sorted(list(set(fds_manual_latex)))

    return fds_manual_latex


def test_all_fds_input_parameters_in_a_list():
    assert len(all_fds_input_parameters_in_a_list()) == 652


def fds2list(fds_script: str, default_param_dict: dict = None):

    res = re.findall(r'&[\s\S]*?/', fds_script)

    list_from_fds = list()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

        if default_param_dict is None:
            dict_from_fds_one_line = dict()
        else:
            dict_from_fds_one_line = copy.copy(default_param_dict)

        group_param_val = fds2dict_parameterise_single_fds_command(v)

        dict_from_fds_one_line['_GROUP'] = group_param_val[0]

        for j in list(range(len(group_param_val)))[1::2]:
            dict_from_fds_one_line[group_param_val[j]] = group_param_val[j+1]

        list_from_fds.append(dict_from_fds_one_line)

    return list_from_fds


def fds2list2(fds_script: str, default_param_list: list):

    res = re.findall(r'&[\s\S]*?/', fds_script)

    list_from_fds = list()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

        list_from_fds_one_line = [None] * len(default_param_list)

        group_param_val = fds2dict_parameterise_single_fds_command(v)

        list_from_fds_one_line[default_param_list.index('_GROUP')] = group_param_val[0]

        for j in list(range(len(group_param_val)))[1::2]:
            if '(' in group_param_val[j]:
                continue
            list_from_fds_one_line[default_param_list.index(group_param_val[j])] = group_param_val[j+1]

        list_from_fds.append(list_from_fds_one_line)

    return list_from_fds


def fds2dict_parameterise_single_fds_command(line: str):
    """Converts a single FDS command in to a list [group_name, parameter1, value1, parameter2, value2, ...]

    :param line: a string contains only one FDS command.
    :return: a list in [group_name, parameter1, value1, parameter2, value2, ...]
    """

    # CHECK IF THE LINE IS FULLY ENCLOSED IN `&` AND `/`
    # ==================================================
    line = re.sub(r"[\n\r]", "", line)
    line = line.strip()
    line = re.findall(r"^&(.+)/", line)
    if len(line) == 0:
        return None
    elif len(line) > 1:
        raise ValueError("Multiple lines of command found.")
    else:
        line = line[0]
    # EXTRACT GROUP NAME
    # ==================

    group_name = re.findall(r"^(\w+) ", line)
    if len(group_name) > 1:
        raise ValueError('Multiple group names found, only 1 expected: ', group_name)
    elif len(group_name) == 0:
        raise ValueError('No group name found.')
    else:
        group_name = group_name[0]
    # remove group_name from the line
    # line = line.replace(group_name, "").strip()
    line = re.sub(r"^\w+ ", "", line)

    # SPLIT TO [parameter, value, parameter, value ...]
    # =================================================

    line = re.split(r"(\w+[\(\),\d]* *= *)", line)
    line = list(filter(None, line))
    rep = re.compile(r"[=,]$")
    for i, v in enumerate(line):
        v = v.strip()
        v = rep.sub("", v)
        line[i] = v
    if len(line) == 0:
        warnings.warn("No parameters found.")
    elif len(line) % 2 != 0:
        raise ValueError("Not always in `parameter, value` pairs.")

    return [group_name] + line


def test_fds2dict_parameterise_single_fds_command():
    from fdspy.preprocessor.fds2dict import fds2dict_parameterise_single_fds_command as ff

    def fff(line_):
        line_ = fds2dict_parameterise_single_fds_command(line_)
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
          AGE=60.0/"""
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


if __name__ == '__main__':

    import pprint
    from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_RIU_MOE1

    # out = fds2list(EXAMPLE_FDS_SCRIPT_RIU_MOE1, {i: None for i in all_fds_input_parameters_in_a_list()})
    # pprint.pprint(out, indent=1, width=80)

    out = fds2list2(EXAMPLE_FDS_SCRIPT_RIU_MOE1, ['_GROUP']+all_fds_input_parameters_in_a_list())
    # pprint.pprint(out, indent=1, width=80)
    out = {i:v for i,v in enumerate(out)}
    import pandas as pd
    out2 = pd.DataFrame.from_dict(out, orient='index', columns=['_GROUP']+all_fds_input_parameters_in_a_list())
    pprint.pprint(out2[out2['_GROUP']=='RAMP'].dropna(axis=1))

    # test_fds_groups_in_a_list()
    # test_fds2dict_parameterise_single_fds_command()

# -*- coding: utf-8 -*-

"""
LIMITATIONS
===========

FDS parameters with array feature are ignored.
i.e. MALT(1,1), anything parameter followed with (#).

"""

import re
import copy
import warnings
import pandas as pd


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


def fds2list3(fds_script: str, default_fds_param_list: list = None):

    fds_command_list = re.findall(r'&[\s\S]*?/', fds_script)

    # MAKE A LIST OF PARAMETER NAMES (i.e. ALL POSSIBLE FDS PARAMETERS)
    # =================================================================

    fds_command_parameterised_list = list()
    if default_fds_param_list is None:
        fds_param_list_all = list()
        for i in fds_command_list:
            fds_group_param_val = fds2dict_parameterise_single_fds_command(i)
            fds_command_parameterised_list.append(fds_group_param_val)
            for j in list(range(len(fds_group_param_val)))[1::2]:
                if '(' in fds_group_param_val[j]:
                    continue
                fds_param_list_all.extend([fds_group_param_val[j]])
        fds_param_list_all += ['_GROUP']
        fds_param_list_all = sorted(list(set(fds_param_list_all)))
    else:
        fds_param_list_all = copy.copy(default_fds_param_list)
        fds_command_parameterised_list = [fds2dict_parameterise_single_fds_command(i) for i in fds_command_list]

    #

    fds_param_list_out = list()  # to store all parameterised fds commands.

    # to check length
    if len(fds_command_list) != len(fds_command_parameterised_list):
        raise ValueError("Length of `fds_command_list` and `fds_command_parameterised_list` not equal.")

    for i, v in enumerate(fds_command_list):

        fds_group_param_val = fds_command_parameterised_list[i]

        # to work out parameterised fds command (single line) in one-hot format.
        fds_parameterised_liner = [None] * len(fds_param_list_all)
        fds_parameterised_liner[fds_param_list_all.index('_GROUP')] = fds_group_param_val[0]
        for j in list(range(len(fds_group_param_val)))[1::2]:
            if '(' in fds_group_param_val[j]:  # ignore array format FDS parameters, i.e. MALT(1,1)
                continue
            fds_parameterised_liner[fds_param_list_all.index(fds_group_param_val[j])] = fds_group_param_val[j+1]

        fds_param_list_out.append(fds_parameterised_liner)

    return fds_param_list_out, fds_param_list_all


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


def test_fds2list3():
    import pprint

    import pandas as pd
    from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_RIU_MOE1
    l0, l1 = fds2list3(EXAMPLE_FDS_SCRIPT_RIU_MOE1)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient='index', columns=l1)
    pprint.pprint(df[df['_GROUP'] == 'RAMP'].dropna(axis=1))


def test_fds2list2():

    import pprint
    import pandas as pd
    from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_RIU_MOE1
    out = fds2list2(EXAMPLE_FDS_SCRIPT_RIU_MOE1, ['_GROUP']+all_fds_input_parameters_in_a_list())
    # pprint.pprint(out, indent=1, width=80)
    out = {i: v for i, v in enumerate(out)}
    out2 = pd.DataFrame.from_dict(out, orient='index', columns=['_GROUP']+all_fds_input_parameters_in_a_list())
    pprint.pprint(out2[out2['_GROUP'] == 'RAMP'].dropna(axis=1))


def fds_analyser_hrr(df: pd.DataFrame):

    """&SURF ID='BURNER 1MW 1.2m',
      COLOR='RED',
      HRRPUA=510.0,
      TAU_Q=-288.0,
      PART_ID='Tracer',
      DT_INSERT=0.5/"""

    a = df[df['HRRPUA']!=None]

    pass


def fds_analyser_slc(df: pd.DataFrame):
    # total no. of slc
    pass


def fds_analyser_general(df: pd.DataFrame):
    # no. of commands
    # no. of group
    # no. of parameters (inc. commands)
    pass


def fds_analyser_fire():
    pass


if __name__ == '__main__':

    test_fds2list3()

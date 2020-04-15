import copy
import re

import pandas as pd


def fds2df(fds_script: str) -> pd.DataFrame:
    """Converts FDS script to a pandas DataFrame object containing parameterised FDS script for ease processing.

    :param fds_script: FDS script string to be analysed and parameterised.
    :return: a pandas DataFrame object containing parameterised FDS script.
    """

    l0, l1 = fds2list(fds_script)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient="index", columns=l1)

    return df


def fds2list(fds_script: str, default_fds_param_list: list = None):

    fds_command_list = re.findall(r"&[\s\S]*?/", fds_script)

    # WORK OUT A LIST OF ALL EXISTED FDS PARAMETER NAMES
    # ==================================================
    fds_param_list_all = list()
    fds_command_parameterised_list = list()
    if default_fds_param_list is None:
        for i in fds_command_list:
            fds_group_param_val = fds2dict_parameterise_single_fds_command(i)
            fds_command_parameterised_list.append(fds_group_param_val)
            for j in list(range(len(fds_group_param_val)))[1::2]:
                if "(" in fds_group_param_val[j]:
                    continue
                fds_param_list_all.extend([fds_group_param_val[j]])
        fds_param_list_all += ["_GROUP"]
        fds_param_list_all = sorted(list(set(fds_param_list_all)))
    else:
        fds_param_list_all = copy.copy(default_fds_param_list)
        fds_command_parameterised_list = [
            fds2dict_parameterise_single_fds_command(i) for i in fds_command_list
        ]

    fds_param_list_out = list()  # to store all parameterised fds commands.

    # to check length
    if len(fds_command_list) != len(fds_command_parameterised_list):
        raise ValueError(
            "Length of `fds_command_list` and `fds_command_parameterised_list` not equal."
        )

    for i, v in enumerate(fds_command_list):

        fds_group_param_val = fds_command_parameterised_list[i]

        # to work out parameterised fds command (single line) in one-hot format.
        fds_parameterised_liner = [None] * len(fds_param_list_all)
        fds_parameterised_liner[fds_param_list_all.index("_GROUP")] = fds_group_param_val[0]
        for j in list(range(len(fds_group_param_val)))[1::2]:
            if (
                "(" in fds_group_param_val[j]
            ):  # ignore array format FDS parameters, i.e. MALT(1,1)
                continue
            fds_parameterised_liner[fds_param_list_all.index(fds_group_param_val[j])] = fds_group_param_val[j + 1]

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
        raise ValueError(f"Multiple group names found, only 1 expected: {group_name}")
    elif len(group_name) == 0:
        raise ValueError(f"No group name found. {line}")
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
    if len(line) % 2 != 0:
        raise ValueError("Not always in `parameter, value` pairs.")

    return [group_name] + line

# -*- coding: utf-8 -*-

"""
LIMITATIONS
===========

FDS parameters with array feature are ignored.
i.e. MALT(1,1), anything parameter followed with (#).

Does not support multiple fires, only supports multiple (redundant) SURF group.
&SURF ID='Burner', COLOR='RED', TMP_FRONT=500. HRRPUA=2672., RAMP_Q='Burner_RAMP_Q'/
&OBST XB=49.00,51.00,3.80,4.80,0.00,0.40, SURF_IDS='Burner','Steel pool','Steel pool'/
"""

import re
import copy
import warnings
import pandas as pd
import numpy as np
import plotly
import plotly.express as pex


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
    out = "\n".join(out)
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
    fds_manual_latex = "\n".join(fds_manual_latex)
    # find all possible FDS input parameters
    fds_manual_latex = re.findall(r"\n{ct\s([\w]+)[(\} *,]", fds_manual_latex) + [
        "PBY",
        "PBZ",
        "FYI",
    ]
    # filter fds_manual_latex duplicated and sort all the items
    fds_manual_latex = sorted(list(set(fds_manual_latex)))

    return fds_manual_latex


def fds2list(fds_script: str, default_param_dict: dict = None):

    res = re.findall(r"&[\s\S]*?/", fds_script)

    list_from_fds = list()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

        if default_param_dict is None:
            dict_from_fds_one_line = dict()
        else:
            dict_from_fds_one_line = copy.copy(default_param_dict)

        group_param_val = fds2dict_parameterise_single_fds_command(v)

        dict_from_fds_one_line["_GROUP"] = group_param_val[0]

        for j in list(range(len(group_param_val)))[1::2]:
            dict_from_fds_one_line[group_param_val[j]] = group_param_val[j + 1]

        list_from_fds.append(dict_from_fds_one_line)

    return list_from_fds


def fds2list2(fds_script: str, default_param_list: list):

    res = re.findall(r"&[\s\S]*?/", fds_script)

    list_from_fds = list()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

        list_from_fds_one_line = [None] * len(default_param_list)

        group_param_val = fds2dict_parameterise_single_fds_command(v)

        list_from_fds_one_line[default_param_list.index("_GROUP")] = group_param_val[0]

        for j in list(range(len(group_param_val)))[1::2]:
            if "(" in group_param_val[j]:
                continue
            list_from_fds_one_line[
                default_param_list.index(group_param_val[j])
            ] = group_param_val[j + 1]

        list_from_fds.append(list_from_fds_one_line)

    return list_from_fds


def fds2list3(fds_script: str, default_fds_param_list: list = None):

    fds_command_list = re.findall(r"&[\s\S]*?/", fds_script)

    # MAKE A LIST OF PARAMETER NAMES (i.e. ALL POSSIBLE FDS PARAMETERS)
    # =================================================================

    fds_command_parameterised_list = list()
    if default_fds_param_list is None:
        fds_param_list_all = list()
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

    #

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
        fds_parameterised_liner[
            fds_param_list_all.index("_GROUP")
        ] = fds_group_param_val[0]
        for j in list(range(len(fds_group_param_val)))[1::2]:
            if (
                "(" in fds_group_param_val[j]
            ):  # ignore array format FDS parameters, i.e. MALT(1,1)
                continue
            fds_parameterised_liner[
                fds_param_list_all.index(fds_group_param_val[j])
            ] = fds_group_param_val[j + 1]

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
        raise ValueError("Multiple group names found, only 1 expected: ", group_name)
    elif len(group_name) == 0:
        raise ValueError("No group name found.")
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


def test_fds2list3():
    import pprint
    import pandas as pd
    from fdspy.lib.fds2dict_data import EXAMPLE_FDS_SCRIPT_RIU_MOE1

    l0, l1 = fds2list3(EXAMPLE_FDS_SCRIPT_RIU_MOE1)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient="index", columns=l1)
    pprint.pprint(df[df["_GROUP"] == "RAMP"].dropna(axis=1))


def test_fds2list2():

    import pprint
    import pandas as pd
    from fdspy.lib.fds2dict_data import EXAMPLE_FDS_SCRIPT_RIU_MOE1

    out = fds2list2(
        EXAMPLE_FDS_SCRIPT_RIU_MOE1, ["_GROUP"] + all_fds_input_parameters_in_a_list()
    )
    out = {i: v for i, v in enumerate(out)}
    out2 = pd.DataFrame.from_dict(
        out, orient="index", columns=["_GROUP"] + all_fds_input_parameters_in_a_list()
    )
    pprint.pprint(out2[out2["_GROUP"] == "RAMP"].dropna(axis=1))


def fds_analyser(df: pd.DataFrame):

    # General Info
    # ============

    print(fds_analyser_general(df))

    # HRR curve
    # =========

    fds_analyser_hrr(df).show()


def fds_analyser_hrr(df: pd.DataFrame) -> pex:

    """&SURF ID='BURNER 1MW 1.2m',
      COLOR='RED',
      HRRPUA=510.0,
      TAU_Q=-288.0,
      PART_ID='Tracer',
      DT_INSERT=0.5/"""

    # Filter items with `_GROUP` == `SURF` and has `HRRPUA` value

    df2 = copy.copy(df)
    df2 = df2[df2["_GROUP"] == "SURF"]
    df2 = df2[df2["HRRPUA"].notnull()]
    df2.dropna(axis=1, inplace=True)

    # Make the above a list

    list_dict_surf_hrrpua = list()
    for i, v in df2.iterrows():
        list_dict_surf_hrrpua.append(v.to_dict())

    for dict_surf_hrrpua in list_dict_surf_hrrpua:

        id = dict_surf_hrrpua["ID"].replace('"', "").replace("'", "")
        list_dict_obst = list()

        df3 = copy.copy(df)  # used to filter obst linked to the surf_hrrpua
        df3 = df3[df3["SURF_IDS"].notnull()]
        df3 = df3[df3["SURF_IDS"].str.contains(id)]
        df3.dropna(axis=1, inplace=True)
        for i, v in df3.iterrows():
            dict_obst = v.to_dict()

            if dict_obst["_GROUP"] != "OBST":
                raise ValueError(
                    "Only `SURF` with `HRRPUA` assigned to `OBST` with `SURF_IDS` is supported."
                )

            # Calculate fire area
            # -------------------
            # identify which index the surf is assigned to
            obst_surf_ids = dict_obst["SURF_IDS"]
            i_assigned = -1
            for i_assigned, v_ in enumerate(obst_surf_ids.split(",")):
                if id in v_:
                    break
            if (
                i_assigned == 1 or i_assigned < 0
            ):  # only supports surf assigned to top or bottom
                raise ValueError(
                    "`SURF` with `HRRPUA` can not assigned to sides in `SURF_IDS`."
                )

            # work out area
            x1, x2, y1, y2, z1, z2 = [float(_) for _ in dict_obst["XB"].split(",")]
            dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
            area = dx * dy

            # Calculate HRRPUA
            # ----------------
            hrrpua = float(dict_surf_hrrpua["HRRPUA"])

            # Calculate hrr against time curve
            # --------------------------------
            df4 = copy.copy(df)
            df4 = df4[df4["T_END"].notna()]
            df4.dropna(axis=1, inplace=True)
            time_array = np.arange(0, float(list(df4["T_END"])[0]) + 1, 1)
            hrr_frac_array = None
            hrr_array = None
            if "TAU_Q" in dict_surf_hrrpua.keys():
                tau_q = float(dict_surf_hrrpua["TAU_Q"])
                if tau_q > 0:
                    hrr_frac_array = np.tanh(time_array / tau_q)
                elif tau_q < 0:
                    hrr_frac_array = (time_array / tau_q) ** 2
                else:
                    raise ValueError("TAU_Q is zero, not good.")
                hrr_frac_array[hrr_frac_array > 1] = 1
                hrr_array = hrr_frac_array * area * hrrpua
            elif "RAMP_Q" in dict_surf_hrrpua.keys():
                ramp_q = dict_surf_hrrpua["RAMP_Q"]

                df5 = df[df["_GROUP"] == "RAMP"]
                df5 = df5[df5["ID"] == ramp_q]
                df5 = df5.dropna(axis=1)

                time_raw = df5["T"].astype(float).values
                frac_raw = df5["F"].astype(float).values
                frac_raw = frac_raw[np.argsort(time_raw)]
                time_raw = np.sort(time_raw)

                hrr_frac_array = np.interp(time_array, time_raw, frac_raw)
                hrr_array = hrr_frac_array * area * hrrpua
            else:
                raise NotImplemented("Only TAU_Q and RAMP_Q are currently supported.")

    fig = pex.line(x=time_array, y=hrr_array, labels=dict(x="Time [s]", y="HRR [kW]"))

    return fig


def fds_analyser_general(df: pd.DataFrame):
    import collections

    sf = "{:<40.40} {}"  # format
    d = collections.OrderedDict()  # to collect results statistics

    d["total number of commands"] = len(df)
    d["total number of unique groups"] = len(list(set(list(df["_GROUP"]))))
    d["total number of unique parameters"] = len(df.columns) - 1
    d["total number of slices"] = len(df[df["_GROUP"] == "SLCF"])

    return "\n".join([sf.format(i, v) for i, v in d.items()])


def main_cli(filepath_fds: str):

    with open(filepath_fds, "r") as f:
        fds_script = f.read()

    main(fds_script)


def main(fds_script: str):

    import pandas as pd

    l0, l1 = fds2list3(fds_script)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient="index", columns=l1)

    fds_analyser(df)


if __name__ == "__main__":

    from fdspy.lib.fds2dict_data import EXAMPLE_FDS_SCRIPT_MALTHOUSE_FF1

    main(EXAMPLE_FDS_SCRIPT_MALTHOUSE_FF1)

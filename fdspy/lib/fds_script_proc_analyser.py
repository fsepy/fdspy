# -*- coding: utf-8 -*-

import copy

import os
import collections
import numpy as np
import pandas as pd
import plotly.express as pex
from fdspy.lib.fds_script_proc_decoder import *

"""
LIMITATIONS
===========

FDS parameters with array feature are ignored.
i.e. MALT(1,1), anything parameter followed with (#).

Does not support multiple fires, only supports multiple (redundant) SURF group.
&SURF ID='Burner', COLOR='RED', TMP_FRONT=500. HRRPUA=2672., RAMP_Q='Burner_RAMP_Q'/
&OBST XB=49.00,51.00,3.80,4.80,0.00,0.40, SURF_IDS='Burner','Steel pool','Steel pool'/
"""


def fds_analyser(df: pd.DataFrame) -> dict:

    std_out_str = list()

    # General Info
    # ============
    std_out_str.append("-" * 40 + "\n")
    std_out_str.append("GENERAL STATISTICS\n")
    std_out_str.append("-" * 40 + "\n")
    std_out_str.append(fds_analyser_general(df))

    # Mesh statistics
    # ===============
    std_out_str.append("-" * 40 + "\n")
    std_out_str.append("MESH STATISTICS\n")
    std_out_str.append("-" * 40 + "\n")
    std_out_str.append(fds_analyser_mesh(df))

    # SLCF statistics
    # ===============

    std_out_str.append("-" * 40 + "\n")
    std_out_str.append("SLCF STATISTICS\n")
    std_out_str.append("-" * 40 + "\n")
    std_out_str.append(fds_analyser_slcf(df))

    # HRR curve
    # =========

    fig_hrr = fds_analyser_hrr(df)

    dict_out = {"str": "".join(std_out_str), "fig_hrr": fig_hrr}

    return dict_out


def fds_analyser_general(df: pd.DataFrame):
    sf = "{:<40.40} {}"  # format
    d = collections.OrderedDict()  # to collect results statistics

    d["command count"] = len(df)
    d["unique group count"] = len(list(set(list(df["_GROUP"]))))
    d["unique parameter count"] = len(df.columns) - 1
    d["simulation duration"] = df["T_END"].dropna().values[0]

    return "\n".join([sf.format(i, v) for i, v in d.items()]) + "\n"


def fds_analyser_mesh(df: pd.DataFrame):
    sf = "{:<40.40} {}"  # format
    d = collections.OrderedDict()  # to collect results statistics

    df1 = df[df["_GROUP"] == "MESH"]
    df1 = df1.dropna(axis=1, inplace=False)

    count_cell = 0
    count_mesh = 0
    length_mesh = 0
    for i, v in df1.iterrows():
        v = v.to_dict()
        ijk = [float(j) for j in v["IJK"].split(",")]
        x1, x2, y1, y2, z1, z2 = [float(j) for j in v["XB"].split(",")]

        count_mesh += 1
        count_cell += np.product(ijk)
        length_mesh += abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)

    count_cell = int(count_cell / 1000)
    d["mesh count"] = f"{count_mesh:d}"
    d["cell count"] = f"{count_cell:,} k"
    d["average cell size"] = str(int(length_mesh / count_cell * 1000)) + " mm"

    return "\n".join([sf.format(i, v) for i, v in d.items()]) + "\n"


def fds_analyser_slcf(df: pd.DataFrame) -> str:
    sf = "{:<40.40} {}"  # format
    d = collections.OrderedDict()  # to collect results statistics

    df1 = copy.copy(df)
    df1 = df1[df1["_GROUP"] == "SLCF"]

    # SLCF counts
    # ===========
    d["slice count"] = len(df1[df1["_GROUP"] == "SLCF"])

    list_quantity = df1["QUANTITY"].values
    for i in sorted(list(set(list_quantity))):
        d[f"SLCF {i} count"] = sum(df1["QUANTITY"] == i)

    # PBX, PBY, PBZ summary
    # =====================
    for i in ["PBX", "PBY", "PBZ"]:
        if i in df1.columns:
            df2 = df1[i].dropna()
            d[f"{i} locations"] = ", ".join(sorted(list(set(df2.values))))
        else:
            d[f"{i} locations"] = "None"

    return "\n".join([sf.format(i, v) for i, v in d.items()]) + "\n"


def fds_analyser_hrr(df: pd.DataFrame) -> pex:
    # GET A LIST OF SURF WITH HRRPUA COMPONENT
    # ========================================
    df1 = copy.copy(df)
    df1 = df1[df1["_GROUP"] == "SURF"]
    df1 = df1[df1["HRRPUA"].notnull()]
    df1.dropna(axis=1, inplace=True)

    list_surfs = list()
    for i, v in df1.iterrows():
        list_surfs.append(v.to_dict())

    # GET A LIST OF OBST/VENT WHOS SURF_ID/SURF_IDS/SURF_ID6 IS ASSOCIATED WITH THE `list_surfs`
    # ==========================================================================================
    list_obst_with_surf_details = list()

    for dict_surf in list_surfs:
        dict_surf_ = copy.copy(dict_surf)  # for inject into OBST/VENT dict
        dict_surf_.pop('_GROUP', None)
        dict_surf_.pop('ID', None)
        id = dict_surf["ID"].replace('"', "").replace("'", "")
        df1 = copy.copy(df)  # used to filter obst linked to the surf_hrrpua
        for k in ['SURF_IDS', 'SURF_ID', 'SURF_ID6']:
            try:
                df2 = df1[df1[k].notna()]
                df2 = df2[df2[k].str.contains(id)]
                df2.dropna(axis=1, how='all', inplace=True)
                for i, v in df2.iterrows():
                    v = v.to_dict()
                    v.pop('ID', None)
                    v.update(dict_surf_)
                    list_obst_with_surf_details.append(v)
            except KeyError:
                pass

    for dict_obst_with_surf_details in list_obst_with_surf_details:
        dict_obst = dict_obst_with_surf_details

        # Calculate fire area
        # -------------------
        x1, x2, y1, y2, z1, z2 = [float(_) for _ in dict_obst["XB"].split(",")]
        dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
        if dict_obst["_GROUP"] != "OBST":
            area = dx * dy
        elif dict_obst["_GROUP"] != "VENT":
            area = max([dx * dy, dy * dz, dz * dx])
        else:
            raise ValueError('Fire should be assigned to OBST or VENT.')

        # Calculate HRRPUA
        # ----------------
        hrrpua = float(dict_obst["HRRPUA"])

        # Calculate hrr against time curve
        # --------------------------------
        # yields `time_array`, `hrr_frac_array` and `hrr_array`
        time_array = np.arange(0, float(df["T_END"].dropna().values[0]) + 1, 1)
        if "TAU_Q" in dict_obst.keys():
            tau_q = float(dict_obst["TAU_Q"])
            if tau_q > 0:
                hrr_frac_array = np.tanh(time_array / tau_q)
            elif tau_q < 0:
                hrr_frac_array = (time_array / tau_q) ** 2
            else:
                raise ValueError("TAU_Q is zero, not good.")
            hrr_frac_array[hrr_frac_array > 1] = 1
            hrr_array = hrr_frac_array * area * hrrpua
        elif "RAMP_Q" in dict_obst.keys():
            ramp_q = dict_obst["RAMP_Q"]

            df5 = df[df["_GROUP"] == "RAMP"]
            df5 = df5[df5["ID"] == ramp_q]
            df5 = df5.dropna(axis=1)

            time_raw = df5["T"].astype(float).values
            frac_raw = df5["F"].astype(float).values
            frac_raw = frac_raw[np.argsort(time_raw)]
            time_raw = np.sort(time_raw)

            hrr_frac_array = np.interp(time_array, time_raw, frac_raw)
            hrr_array = hrr_frac_array * area * hrrpua
        elif (
            "RAMP_T" in dict_obst.keys()
            or "RAMP_V" in dict_obst.keys()
        ):
            raise NotImplemented("Only TAU_Q and RAMP_Q are currently supported.")
        else:
            hrr_frac_array = np.full_like(time_array, fill_value=1.0, dtype=float)
            hrr_array = hrr_frac_array * area * hrrpua

    fig = pex.line(
        x=time_array,
        y=hrr_array,
        labels=dict(x="Time [s]", y="HRR [kW]"),
        height=None,
        width=800,
    )

    return fig


def main_cli(filepath_fds: str):
    """CLI main function, the only difference is main_cli takes a file path rather than FDS script."""
    with open(filepath_fds, "r") as f:
        fds_script = f.read()

    dict_out = main(fds_script)

    header_str = "=" * 40 + "\n"
    header_str += os.path.basename(filepath_fds) + "\n"
    header_str += "=" * 40 + "\n"

    dict_out["str"] = header_str + dict_out["str"]

    return dict_out


def main(fds_script: str):
    """"""
    l0, l1 = fds2list3(fds_script)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient="index", columns=l1)
    return fds_analyser(df)


if __name__ == "__main__":

    # main(EXAMPLE_FDS_SCRIPT_MALTHOUSE_FF1)

    main_cli(
        r"C:\Users\ian\Google Drive\projects\fdspy\ofr_scripts\POST_Google_Stage_4_CFD_ScenarioD.fds"
    )

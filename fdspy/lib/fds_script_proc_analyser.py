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
        try:
            df3 = df3[df3["SURF_IDS"].notnull()]
            df3 = df3[df3["SURF_IDS"].str.contains(id)]
        except KeyError:
            try:
                df3 = df3[df3["SURF_ID"].notnull()]
                df3 = df3[df3["SURF_ID"].str.contains(id)]
            except KeyError:
                pass
        df3.dropna(axis=1, inplace=True)
        for i, v in df3.iterrows():
            dict_obst = v.to_dict()

            # Calculate fire area
            # -------------------
            # identify which index the surf is assigned to
            try:
                obst_surf_ids = dict_obst["SURF_IDS"]
            except KeyError:
                obst_surf_ids = dict_obst["SURF_ID"]

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
            if dict_obst["_GROUP"] != "OBST":
                area = dx * dy
            elif dict_obst["_GROUP"] != "VENT":
                area = max([dx * dy, dy * dz, dz * dx])

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
            elif (
                "RAMP_T" in dict_surf_hrrpua.keys()
                or "RAMP_V" in dict_surf_hrrpua.keys()
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

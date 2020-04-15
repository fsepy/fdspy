# -*- coding: utf-8 -*-

import collections
import logging
import os

import numpy as np
import plotly.express as pex

from fdspy.lib.fds_script_proc_decoder import *

logger = logging.getLogger('cli')

"""
LIMITATIONS
===========

FDS parameters with array feature are ignored.
i.e. MALT(1,1), anything parameter followed with (#).

Does not support multiple fires.
"""


def fds_analyser(df: pd.DataFrame):

    std_out_str = list()

    def sf_1_1(s1, s2):
        sf = "{:<40.40} {}"  # format
        if isinstance(s2, int):
            return "{:<40.40} {:d}".format(s1, s2)
        elif isinstance(s2, float):
            return "{:<40.40} {:.2f}".format(s1, s2)
        elif isinstance(s2, list) or isinstance(s2, tuple):
            if isinstance(s2[0], int):
                s2 = ["{:d}".format(i) for i in s2]
                s2 = ', '.join(s2)
                return "{:<40.40} {}".format(s1, s2)
            elif isinstance(s2[0], float):
                s2 = ["{:.2f}".format(i) for i in s2]
                s2 = ', '.join(s2)
                return "{:<40.40} {}".format(s1, s2)
            else:
                s2 = ', '.join(s2)
                return "{:<40.40} {}".format(s1, s2)
        else:
            return "{:<40.40} {}".format(s1, s2)

    # General Info
    # ============
    try:
        std_out_str.append("-" * 40 + "\n")
        std_out_str.append("GENERAL STATISTICS\n")
        std_out_str.append("-" * 40 + "\n")
        _ = fds_analyser_general(df)
        _ = "\n".join([sf_1_1(i, v) for i, v in _.items()]) + "\n"
        std_out_str.append(_)
    except Exception as e:
        logger.error(f'Failed to produce GENERAL STATISTICS. Error {str(e)}')

    # Mesh statistics
    # ===============
    try:
        std_out_str.append("-" * 40 + "\n")
        std_out_str.append("MESH STATISTICS\n")
        std_out_str.append("-" * 40 + "\n")
        _ = fds_analyser_mesh(df)
        _ = "\n".join([sf_1_1(i, v) for i, v in _.items()]) + "\n"
        std_out_str.append(_)
    except Exception as e:
        logger.error(f'Failed to produce MESH STATISTICS. Error {str(e)}')

    # SLCF statistics
    # ===============
    try:
        std_out_str.append("-" * 40 + "\n")
        std_out_str.append("SLCF STATISTICS\n")
        std_out_str.append("-" * 40 + "\n")
        _ = fds_analyser_slcf(df)
        _ = "\n".join([sf_1_1(i, v) for i, v in _.items()]) + "\n"
        std_out_str.append(_)
    except Exception as e:
        logger.error(f'Failed to produce SLCF STATISTICS. Error {str(e)}')

    # HRR statistics
    # ==============
    fig_hrr = None
    try:
        std_out_str.append("-" * 40 + "\n")
        std_out_str.append("HRR STATISTICS\n")
        std_out_str.append("-" * 40 + "\n")
        _ = fds_analyser_hrr(df)
        fig_hrr = fds_analyser_hrr_fig(_)
        _.pop('time_array')
        _.pop('hrr_array')
        std_out_str.append("\n".join([sf_1_1(i, v) for i, v in _.items()]) + "\n")
    except Exception as e:
        logger.error(f'Failed to produce HRR STATISTICS. Error {str(e)}')

    # Packing up results and return
    # =============================
    dict_out = {"str": "".join(std_out_str), "fig_hrr": fig_hrr}

    return dict_out


def fds_analyser_general(df: pd.DataFrame) -> dict:
    d = collections.OrderedDict()  # to collect results statistics

    d["command count"] = len(df)
    # d["unique group count"] = len(list(set(list(df["_GROUP"]))))
    # d["unique parameter count"] = len(df.columns) - 1
    d["sim. duration"] = df["T_END"].dropna().values[0]

    # work out number of mpi processes
    try:
        d['mpi process'] = len(set(df['MPI_PROCESS'].dropna().values))
    except KeyError:
        d['mpi process'] = 1

    return d


def fds_analyser_mesh(df: pd.DataFrame) -> dict:
    d = collections.OrderedDict()  # to collect results statistics

    d_star = fds_analyser_hrr(df)['D*']

    df1 = df[df["_GROUP"] == "MESH"]
    df1 = df1.dropna(axis=1, inplace=False)

    cell_count_i = list()
    cell_size_i = list()
    volume_i = list()
    for i, v in df1.iterrows():
        v = v.to_dict()
        ii, jj, kk = [float(j) for j in v["IJK"].split(",")]
        x1, x2, y1, y2, z1, z2 = [float(j) for j in v["XB"].split(",")]

        cell_count_i.append(ii*jj*kk)
        cell_size_i.append([abs(x2-x1)/ii, abs(y2-y1)/jj, abs(z2-z1)/kk])
        volume_i.append(abs(x1 - x2) * abs(y1 - y2) * abs(z1 - z2))

    d["mesh count"] = "{:d}".format(len(cell_count_i))
    d["cell count"] = "{:,d} k".format(int(np.sum(cell_count_i)/1000))
    d["ave. cell size"] = '{:.0f} mm'.format(((np.sum(volume_i) / np.sum(cell_count_i)) ** (1/3)) * 1000)

    for i, cell_count in enumerate(cell_count_i):
        cell_size = cell_size_i[i]
        d[f"mesh {i:d} cell size"] = cell_size
        d[f"mesh {i:d} D*/dx (max., min.)"] = [d_star/np.max(cell_size), d_star/np.min(cell_size)]

    return d


def fds_analyser_slcf(df: pd.DataFrame) -> dict:
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
            # d[f"{i} locations"] = "None"
            pass

    return d


def fds_analyser_hrr(df: pd.DataFrame):
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

    return {
        'time_array': time_array,
        'hrr_array': hrr_array,
        'peak HRR': np.max(hrr_array),
        'peak HRR time': np.min(time_array[np.argmax(hrr_array)]),
        'D*': (np.max(hrr_array)/(1.204*1.005*293*9.81)) ** (2/5)
    }


def fds_analyser_hrr_fig(data_dict: dict):

    fig = pex.line(
        x=data_dict['time_array'],
        y=data_dict['hrr_array'],
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
    l0, l1 = fds2list(fds_script)
    d = {i: v for i, v in enumerate(l0)}
    df = pd.DataFrame.from_dict(d, orient="index", columns=l1)

    return fds_analyser(df)


if __name__ == "__main__":

    # main(EXAMPLE_FDS_SCRIPT_MALTHOUSE_FF1)

    main_cli(
        r"C:\Users\IanFu\Desktop\fdspy test\moe2.fds"
    )

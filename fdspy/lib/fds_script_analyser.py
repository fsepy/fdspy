import collections
import copy
import logging
import re

import numpy as np
import pandas as pd

from fdspy.lib.asciiplot import AsciiPlot

logger = logging.getLogger('cli')


class Model:
    def __init__(self, fds_raw: str):
        self.__fds_raw = None
        self.__fds_df = None

        self.fds_raw = fds_raw
        self.fds_df = self._fds2df(fds_raw)

    @staticmethod
    def _fds2df(fds_raw: str) -> pd.DataFrame:
        """Converts FDS script to a pandas DataFrame object containing parameterised FDS script for ease processing.

        :param fds_list: FDS script string to be analysed and parameterised.
        :return: a pandas DataFrame object containing parameterised FDS script.
        """

        l0, l1 = Model._fds2list(fds_raw)
        d = {i: v for i, v in enumerate(l0)}
        df = pd.DataFrame.from_dict(d, orient="index", columns=l1)

        return df

    @staticmethod
    def _fds2list(fds_raw: str):

        # parse all fds commands to a list, i.e. (command1, command2, command3, ...)
        fds_command_list: list = re.findall(r"&[\s\S]*?/", fds_raw)

        # ================================================
        # Work out all group names that used in the script
        # ================================================
        fds_param_list_all = list()  # to store group names
        fds_command_parameterised_list = list()  # to store command lists
        for i in fds_command_list:
            fds_group_param_val = Model._fds2list_single_line(i)
            fds_command_parameterised_list.append(fds_group_param_val)
            for j in list(range(len(fds_group_param_val)))[1::2]:
                if "(" in fds_group_param_val[j]:
                    continue
                fds_param_list_all.extend([fds_group_param_val[j]])
        fds_param_list_all = sorted(list(set(fds_param_list_all + ['_GROUP'])))

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

    @staticmethod
    def _fds2list_single_line(line: str):
        """Converts a single FDS command in to a list [group_name, parameter1, value1, parameter2, value2, ...]

        :param line: a string contains only one FDS command.
        :return: a list in [group_name, parameter1, value1, parameter2, value2, ...]
        """

        # =======================
        # clean input fds command
        # =======================
        # check command is enclosed within `&` and `/`
        line = re.sub(r"[\n\r]", "", line)  # remove new line characters
        line = line.strip()
        line = re.findall(r"^&(.+)/", line)
        if len(line) == 0:
            # no command found
            return None
        elif len(line) > 1:
            # multiple fds command is supplied, this function only works for single fds command line
            raise ValueError(
                "Multiple lines of command found. "
                "A single line of FDS command should be enclosed within `&` and `/`"
            )
        else:
            # get the command main content only, i.e. this will remove comments or anything outside the `&` and `/`
            line = line[0]

        # ==================
        # Extract group name
        # ==================
        group_name = re.findall(r"^(\w+) ", line)
        if len(group_name) > 1:
            raise ValueError(f"Multiple group names found, only 1 expected: {group_name}")
        elif len(group_name) == 0:
            raise ValueError(f"No group name found. {line}")
        else:
            group_name = group_name[0]

        line = re.sub(r"^\w+ ", "", line)  # remove group_name from the line

        # =============================================================================
        # Rearrange the fds command line into [parameter, value, parameter, value, ...]
        # =============================================================================
        line = re.split(r"(\w+[\(\),\d]* *= *)", line)
        line = list(filter(None, line))
        rep = re.compile(r"[=,]$")
        for i, v in enumerate(line):
            v = v.strip()
            v = rep.sub("", v)
            line[i] = v
        if len(line) % 2 != 0:
            raise ValueError("Attempted to rearrange fds command, not always in `parameter, value` pairs.")

        return [group_name] + line

    @property
    def fds_raw(self):
        return self.__fds_raw

    @fds_raw.setter
    def fds_raw(self, v: str):
        self.__fds_raw = v

    @property
    def fds_df(self):
        return self.__fds_df

    @fds_df.setter
    def fds_df(self, v: pd.DataFrame):
        self.__fds_df = v


class ModelAnalyser(Model):
    def __init__(self, fds_raw: str, print_width: int = 80):
        super().__init__(fds_raw=fds_raw)

        self.__hrr_x = None
        self.__hrr_y = None
        self.__hrr_d_star = None
        self.__print_width = None

        self.hrr_x, self.hrr_y, self.hrr_d_star = self._heat_release_rate(self.fds_df)
        self.print_width = print_width

    def hrr_plot(self, size: tuple = (80, 20)):

        if len(self.hrr_x) == 0:
            logger.info('No fire detected (currently unable to analyse MLR fire)')
            return ''

        aplot = AsciiPlot(size=size)

        s_start = 'HRR PLOT START'
        s_start = '=' * int((self.print_width - len(s_start)) * 0.5) + s_start
        s_start = s_start + '=' * (self.print_width - len(s_start))

        s_content = aplot.plot(self.hrr_x, self.hrr_y).str()

        s_end = 'HRR PLOT END'
        s_end = '=' * int((self.print_width - len(s_end)) * 0.5) + s_end
        s_end = s_end + '=' * (self.print_width - len(s_end))

        return '\n'.join([s_start, s_content, s_end]) + '\n'

    def general(self) -> str:
        df = self.fds_df
        d = collections.OrderedDict()  # to collect results statistics
        d["Command count"] = len(df)
        d["Simulation duration"] = df["T_END"].dropna().values[0]

        # work out number of mpi processes
        try:
            d['MPI process'] = len(set(df['MPI_PROCESS'].dropna().values))
        except KeyError:
            d['MPI process'] = 1

        s_start = self._make_start_end_line('GENERAL STATS START')

        len_key = min([max([len(i) for i in d.keys()]) + 1, int(self.print_width * 0.6)])
        fmt = f'{{:<{len_key:d}.{len_key:d}}}: {{:<{self.print_width - len_key - 2}}}'
        s_content = "\n".join([fmt.format(k, v) for k, v in d.items()])

        s_end = self._make_start_end_line('GENERAL STATS END')

        return '\n'.join([s_start, s_content, s_end]) + '\n'

    def mesh(self) -> str:
        d = collections.OrderedDict()  # to collect results statistics
        df = self.fds_df

        d_star = self.hrr_d_star

        df1 = df[df["_GROUP"] == "MESH"]
        df1 = df1.dropna(axis=1, inplace=False)

        cell_count_i = list()
        cell_size_i = list()
        volume_i = list()
        for i, v in df1.iterrows():
            v = v.to_dict()
            ii, jj, kk = [float(j) for j in v["IJK"].split(",")]
            x1, x2, y1, y2, z1, z2 = [float(j) for j in v["XB"].split(",")]

            cell_count_i.append(ii * jj * kk)
            cell_size_i.append([abs(x2 - x1) / ii, abs(y2 - y1) / jj, abs(z2 - z1) / kk])
            volume_i.append(abs(x1 - x2) * abs(y1 - y2) * abs(z1 - z2))

        d["Mesh count"] = "{:d}".format(len(cell_count_i))
        d["Cell count"] = "{:,d} k".format(int(np.sum(cell_count_i) / 1000))
        d["Average cell size"] = '{:.0f} mm'.format(((np.sum(volume_i) / np.sum(cell_count_i)) ** (1 / 3)) * 1000)

        for i, cell_count in enumerate(cell_count_i):
            cell_size = cell_size_i[i]
            d[f"Mesh {i:d} cell size"] = ', '.join([f'{j:.3f}'.strip('0').strip('.') for j in cell_size])
            d[f"Mesh {i:d} D*/dx (max., min.)"] = f'{d_star / np.max(cell_size):.3f}, {d_star / np.min(cell_size):.3f}'

        # =====================
        # prepare output string
        # =====================
        s_start = self._make_start_end_line('MESH STATS START')
        len_key = min([max([len(i) for i in d.keys()]) + 1, int(self.print_width * 0.6)])
        fmt = f'{{:<{len_key:d}}}: {{:<{self.print_width - len_key - 2}}}'
        s_content = "\n".join([fmt.format(k, v) for k, v in d.items()])

        s_end = self._make_start_end_line('MESH STATS END')

        return '\n'.join([s_start, s_content, s_end]) + '\n'

    def slcf(self) -> str:
        d = collections.OrderedDict()  # to collect results statistics
        df = self.fds_df

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

        # =====================
        # prepare output string
        # =====================
        s_start = self._make_start_end_line('MESH STATS START')
        len_key = min([max([len(i) for i in d.keys()]) + 1, int(self.print_width * 0.6)])
        fmt = f'{{:<{len_key:d}}}: {{:<{self.print_width - len_key - 2}}}'
        s_content = "\n".join([fmt.format(k, v) for k, v in d.items()])
        s_end = self._make_start_end_line('MESH STATS END')

        return '\n'.join([s_start, s_content, s_end]) + '\n'

    @staticmethod
    def _heat_release_rate(df: pd.DataFrame):
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
            t = np.arange(0, float(df["T_END"].dropna().values[0]) + 1, 1)
            if "TAU_Q" in dict_obst.keys():
                tau_q = float(dict_obst["TAU_Q"])
                if tau_q > 0:
                    hrr_frac_array = np.tanh(t / tau_q)
                elif tau_q < 0:
                    hrr_frac_array = (t / tau_q) ** 2
                else:
                    raise ValueError("TAU_Q is zero, not good.")
                hrr_frac_array[hrr_frac_array > 1] = 1
                hrr = hrr_frac_array * area * hrrpua
            elif "RAMP_Q" in dict_obst.keys():
                ramp_q = dict_obst["RAMP_Q"]

                df5 = df[df["_GROUP"] == "RAMP"]
                df5 = df5[df5["ID"] == ramp_q]
                df5 = df5.dropna(axis=1)

                time_raw = df5["T"].astype(float).values
                frac_raw = df5["F"].astype(float).values
                frac_raw = frac_raw[np.argsort(time_raw)]
                time_raw = np.sort(time_raw)

                hrr_frac_array = np.interp(t, time_raw, frac_raw)
                hrr = hrr_frac_array * area * hrrpua
            elif (
                    "RAMP_T" in dict_obst.keys()
                    or "RAMP_V" in dict_obst.keys()
            ):
                raise NotImplemented("Only TAU_Q and RAMP_Q are currently supported.")
            else:
                hrr_frac_array = np.full_like(t, fill_value=1.0, dtype=float)
                hrr = hrr_frac_array * area * hrrpua

        d_star = (np.max(hrr) / (1.204 * 1.005 * 293 * 9.81)) ** (2 / 5)

        return t, hrr, d_star

    def _make_start_end_line(self, s: str):
        s = '=' * int((self.print_width - len(s)) * 0.5) + s
        s = s + '=' * (self.print_width - len(s))
        return s

    @property
    def hrr_x(self):
        return self.__hrr_x

    @hrr_x.setter
    def hrr_x(self, v: np.ndarray):
        self.__hrr_x = v

    @property
    def hrr_y(self):
        return self.__hrr_y

    @hrr_y.setter
    def hrr_y(self, v: np.ndarray):
        self.__hrr_y = v

    @property
    def hrr_d_star(self):
        return self.__hrr_d_star

    @hrr_d_star.setter
    def hrr_d_star(self, v: float):
        self.__hrr_d_star = v

    @property
    def print_width(self):
        return self.__print_width

    @print_width.setter
    def print_width(self, v: int):
        assert isinstance(v, int)
        self.__print_width = v


def __test_Model_fds2list_single_line():
    from fdspy.lib.fds_script_proc_analyser import (
        fds2dict_parameterise_single_fds_command as ff,
    )
    ff = Model._fds2list_single_line

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
    assert ff(line)[3] == "RESTRICT_TIME_STEP"


def __test_Model_fds2df():
    from os import path
    import fdspy

    fp_fds = path.join(path.dirname(fdspy.__root_dir__), 'tests', 'fds_scripts', 'general-residential_corridor.fds')

    with open(fp_fds, 'r') as f:
        fds_raw = f.read()

    model = Model(fds_raw)


def __test_ModelAnalyser():
    from os import path
    import fdspy

    fp_fds = path.join(path.dirname(fdspy.__root_dir__), 'tests', 'fds_scripts', 'general-residential_corridor.fds')

    with open(fp_fds, 'r') as f:
        fds_raw = f.read()

    model = ModelAnalyser(fds_raw)

    print(model.hrr_plot(size=(80, 10)))
    print(model.general())
    print(model.mesh())
    print(model.slcf())


if __name__ == '__main__':
    __test_Model_fds2list_single_line()
    __test_Model_fds2df()
    __test_ModelAnalyser()

import collections
import copy

import numpy as np
import pandas as pd

from fdspy import logger
from fdspy.lib.asciiplot import AsciiPlot
from fdspy.lib.fds_script_core import FDSBaseModel


class FDSAnalyserBase:
    @staticmethod
    def general(df_fds: pd.DataFrame) -> dict:
        dict_stats = dict()  # to collect results statistics
        dict_stats["Command count"] = len(df_fds)
        dict_stats["Simulation duration"] = df_fds["T_END"].dropna().values[0]

        # work out number of mpi processes
        try:
            dict_stats['MPI process'] = len(set(df_fds['MPI_PROCESS'].dropna().values))
        except KeyError:
            dict_stats['MPI process'] = 1

        return dict_stats

    @staticmethod
    def mesh(df_fds: pd.DataFrame, d_star: float) -> dict:
        d = collections.OrderedDict()  # to collect results statistics

        df_fds_mesh = df_fds[df_fds["_GROUP"] == "MESH"]
        df_fds_mesh = df_fds_mesh.dropna(axis=1, inplace=False)

        cell_count_i, cell_size_i, volume_i = list(), list(), list()
        for i, v in df_fds_mesh.iterrows():
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

        return d

    @staticmethod
    def slcf(df_fds: pd.DataFrame) -> dict:
        dict_stats = dict()  # to collect results statistics

        df_fds_slcf = df_fds.copy()
        df_fds_slcf = df_fds_slcf[df_fds_slcf["_GROUP"] == "SLCF"]

        # SLCF counts
        # ===========
        dict_stats["slice count"] = len(df_fds_slcf[df_fds_slcf["_GROUP"] == "SLCF"])

        list_quantity = df_fds_slcf["QUANTITY"].values
        for i in sorted(list(set(list_quantity))):
            dict_stats[f"SLCF {i} count"] = sum(df_fds_slcf["QUANTITY"] == i)

        # PBX, PBY, PBZ summary
        # =====================
        for i in ["PBX", "PBY", "PBZ"]:
            if i in df_fds_slcf.columns:
                df2 = df_fds_slcf[i].dropna()
                dict_stats[f"{i} locations"] = ", ".join(sorted(list(set(df2.values))))
            else:
                # d[f"{i} locations"] = "None"
                pass

        return dict_stats

    @staticmethod
    def burner(df_fds: pd.DataFrame, figsize=(80, 20)):
        dict_stats = dict()
        # ================
        # Find all burners
        # ================
        df_fds_hrrpua = df_fds[df_fds["_GROUP"] == "SURF"]
        df_fds_hrrpua = df_fds_hrrpua[df_fds_hrrpua["HRRPUA"].notnull()]
        df_fds_hrrpua.dropna(axis=1, inplace=True)

        # =====================================
        # Generate time and HRR for all burners
        # =====================================
        # e.g. list_burner_hrr = [(id1, t1, hrr1), (id2, t2, hrr2), ...]
        list_burner_hrr = list()

        # ===================
        # Make time-HRR plots
        # ===================
        aplot = AsciiPlot(size=figsize)
        for i, (id, t, hrr) in enumerate(list_burner_hrr):
            dict_stats[f'Fire {i}-{id}'] = aplot.plot(t, hrr).str() + '\n'

        return dict_stats


class FDSAnalyser(FDSBaseModel):
    def __init__(self, fds_raw: str, print_width: int = 80):
        super().__init__(fds_raw=fds_raw)

        self.__hrr_x = None
        self.__hrr_y = None
        self.__hrr_d_star = None
        self.__print_width = None

        try:
            self.hrr_x, self.hrr_y, self.hrr_d_star = self._heat_release_rate(self.fds_df)
        except Exception as e:
            logger.debug(f'Failed to parse heat release rate, {e}')

        self.print_width = print_width

    def hrr_plot(self, size: tuple = (80, 20)):
        try:
            return self.__hrr_plot(size=size)
        except Exception as e:
            return str(e)

    def __hrr_plot(self, size: tuple = (80, 20)) -> str:

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
        try:
            return self.__general()
        except Exception as e:
            return str(e)

    def __general(self) -> str:
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
        try:
            return self.__mesh()
        except Exception as e:
            return str(e)

    def __mesh(self) -> str:
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
        try:
            return self.__slcf()
        except Exception as e:
            return str(e)

    def __slcf(self) -> str:
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


def _test_FDSAnalyser():
    from os import path
    import fdspy

    fp_fds_list = [
        'general-benchmark_1.fds', 'general-error.fds', 'general-residential_corridor.fds', 'general-room_fire.fds',
        'mesh-0.fds', 'mesh-1.fds', 'mesh-2.fds', 'mesh_16_1m.fds', 'travelling_fire-1cw.fds',
        'travelling_fire-line-1_ignition_origin.fds',
    ]
    for fp_fds in fp_fds_list:
        print(fp_fds)

        fp_fds = path.join(fdspy.__root_dir__, 'tests', 'fds_scripts', fp_fds)

        with open(fp_fds, 'r') as f:
            fds_raw = f.read()

        model = FDSAnalyser(fds_raw)

        model.hrr_plot(size=(80, 10))
        model.general()
        model.mesh()
        model.slcf()
        df = copy.copy(model.fds_df)
        try:
            n_mpi = len(set(df['MPI_PROCESS'].dropna().values))
        except:
            n_mpi = '-1'
        print(n_mpi)


if __name__ == '__main__':
    _test_FDSAnalyser()

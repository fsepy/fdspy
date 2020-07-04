import logging
from os.path import join, dirname
from typing import List, Tuple

import numpy as np
import pandas as pd

from fdspy.lib.fds_script_analyser import FDSBaseModel

logger = logging.getLogger('cli')


class FDSTravellingFireMaker(FDSBaseModel):

    def __init__(self, fds_raw: str):
        super().__init__(fds_raw=fds_raw)

    def make_travelling_fire(
            self,
            id_common_prefix: List[str],
            ignition_origin: List[Tuple[float, float, float]],
            ignition_delay_time: List[float],
            spread_speed: List[float],
            burning_time: List[float]
    ):

        assert len(id_common_prefix) == len(ignition_origin) == len(spread_speed) == len(burning_time)

        fds_df = self.fds_df.copy()

        # check for columns `CTRL_ID`, `FUNCTION_TYPE`, `RAMP_ID`, `LATCH`, `INPUT_ID` `T` and `F`
        # insert to the dataframe if any one of above does not exist
        for i in ['CTRL_ID', 'FUNCTION_TYPE', 'RAMP_ID', 'LATCH', 'INPUT_ID', 'T', 'F']:
            if i not in list(fds_df):
                fds_df[i] = None

        # ------------------
        # Make CTRL and RAMP
        # ------------------
        ctrls, ramps = list(), list()
        for i in range(len(id_common_prefix)):
            ctrls_, ramps_ = self._travelling_fire_make_ctrls_and_ramps(
                fds_df=fds_df,
                id_common_prefix=id_common_prefix[i],
                ignition_origin=ignition_origin[i],
                spread_speed=spread_speed[i],
                ignition_delay_time=ignition_delay_time[i],
                burning_time=burning_time[i]
            )
            ctrls.extend(ctrls_)
            ramps.extend(ramps_)

            fds_df = self._travelling_fire_add_ctrl_id_to_vents(
                fds_df=fds_df,
                id_common_prefix=id_common_prefix[i],
                ignition_origin=ignition_origin[i]
            )

        # ---------
        # Make devc
        # ---------
        x1, x2, y1, y2, z1, z2 = [float(i) for i in fds_df[fds_df['_GROUP'] == 'MESH'].iloc[0]['XB'].split(',')]
        devc = dict(
            _GROUP='DEVC',
            ID=f"'travelling_fire_devc'",
            QUANTITY="'TIME'",
            XYZ=f'{(x1 + x2) / 2:.1f},{(y1 + y2) / 2:.1f},{(z1 + z2) / 2:.1f}'
        )

        # -----------------
        # Combine DataFrame
        # -----------------
        fds_df_new = pd.DataFrame.from_dict(ctrls + ramps + [devc])

        index_insert = np.amax(fds_df.index[fds_df['_GROUP'] == 'MESH'].tolist()) + 1
        fds_df_new_all = pd.concat([fds_df.iloc[0:index_insert], fds_df_new, fds_df.iloc[index_insert::]],
                                   sort=False)
        fds_df_new_all = fds_df_new_all.where(pd.notnull(fds_df_new_all), None)
        fds_df_new_all.reset_index(inplace=True, drop=True)
        logger.debug(fds_df_new_all)

        fds_df = fds_df_new_all

        return self._df2fds(fds_df)

    @staticmethod
    def _travelling_fire_add_ctrl_id_to_vents(
            fds_df: pd.DataFrame,
            id_common_prefix: str,
            ignition_origin: Tuple[float, float, float]
    ):
        for index, row in fds_df[fds_df['_GROUP'] == 'VENT'].iterrows():
            x1, x2, y1, y2, z1, z2 = [float(i) for i in row['XB'].split(',')]  # vent geometry
            x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2  # vent centroid
            # pass if (1) vent is not on the same z-axis plane; (2) id does not contain `id_common_prefix`
            if not abs(z - ignition_origin[2]) <= 1e-4 and id_common_prefix not in row['ID']:
                continue
            id_ = row['ID'].replace('"', '').replace("'", '')  # vent id
            fds_df.iloc[index]['CTRL_ID'] = f"'{id_}'"  # set CTRL_ID for vent

        return fds_df

    @staticmethod
    def _travelling_fire_make_ctrls_and_ramps(
            fds_df: pd.DataFrame,
            id_common_prefix: str,
            ignition_origin: Tuple[float, float, float],
            ignition_delay_time: float,
            spread_speed: float,
            burning_time: float

    ):
        ctrls = list()
        ramps = list()
        for index, row in fds_df[fds_df['_GROUP'] == 'VENT'].iterrows():
            x1, x2, y1, y2, z1, z2 = [float(i) for i in row['XB'].split(',')]  # vent geometry
            x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2  # vent centroid
            id_ = row['ID'].replace('"', '').replace("'", '')  # vent id

            # pass if (1) vent is not on the same z-axis plane; (2) id does not contain `id_common_prefix`
            if not abs(z - ignition_origin[2]) <= 1e-4 or id_common_prefix not in row['ID']:
                continue

            # Calculate start time t1 and end time t2
            d = ((x - ignition_origin[0]) ** 2 + (y - ignition_origin[1]) ** 2) ** 0.5
            t1 = d / spread_speed + ignition_delay_time
            t2 = t1 + burning_time

            # Make CTRL
            ctrls.append(
                dict(
                    _GROUP="CTRL",
                    ID=f"'{id_}'",
                    FUNCTION_TYPE="'CUSTOM'",
                    RAMP_ID=f"'{id_}_RAMP'",
                    LATCH=".FALSE.",
                    INPUT_ID=f"'travelling_fire_devc'",
                )
            )

            # Make RAMP
            for i in range(4):
                ramp = dict(
                    _GROUP="RAMP",
                    ID=f"'{id_}_RAMP'",
                    T=f"{t1 - 0.25 if i == 0 else (t1 + 0.25 if i == 1 else (t2 - 0.25 if i == 2 else t2 + 0.25)):.2f}",
                    F=f"{1 if 1 <= i <= 2 else -1:.1f}",
                )
                ramps.append(ramp)

        return ctrls, ramps


def __test_travelling_fire_line_1_ignition():
    from fdspy import __root_dir__

    fp_fds_raw = join(dirname(__root_dir__), 'tests', 'fds_scripts', 'travelling_fire-line-1_ignition_origin.fds')

    with open(fp_fds_raw, 'r') as f:
        fds_raw = f.read()

    test = FDSTravellingFireMaker(fds_raw=fds_raw)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire'],
        ignition_origin=[(0, 0.5, 0)],
        ignition_delay_time=[0],
        spread_speed=[0.2],
        burning_time=[10]
    )

    from os.path import basename
    with open(join(dirname(fp_fds_raw), basename(fp_fds_raw.replace('.fds', '.out.fds'))), 'w+') as f:
        f.write(res)


def __test_travelling_fire_line_2_ignition():
    from fdspy import __root_dir__

    fp_fds_raw = join(dirname(__root_dir__), 'tests', 'fds_scripts', 'travelling_fire-line-2_ignition_origin.fds')

    with open(fp_fds_raw, 'r') as f:
        fds_raw = f.read()

    test = FDSTravellingFireMaker(fds_raw=fds_raw)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire_a', 'travelling_fire_b'],
        ignition_origin=[(0, 0.5, 0), (5, 0.5, 0)],
        ignition_delay_time=[0, 0],
        spread_speed=[0.2, 0.2],
        burning_time=[10, 10]
    )

    from os.path import basename
    with open(join(dirname(fp_fds_raw), basename(fp_fds_raw.replace('.fds', '.out.fds'))), 'w+') as f:
        f.write(res)


def __test_travelling_fire_1cw():
    from fdspy import __root_dir__

    fp_fds_raw = join(dirname(__root_dir__), 'tests', 'fds_scripts', 'travelling_fire-1cw.fds')

    with open(fp_fds_raw, 'r') as f:
        fds_raw = f.read()

    test = FDSTravellingFireMaker(fds_raw=fds_raw)
    res = test.make_travelling_fire(
        id_common_prefix=['travelling_fire_a', 'travelling_fire_b', 'travelling_fire_c'],
        ignition_origin=[(3.3, 46.7, 3.82), (24.3, 28.7, 3.82), (36.3, 27.7, 3.82)],
        ignition_delay_time=[0, 27.5, 27.5 + 12.0],
        spread_speed=[1, 1, 1],
        burning_time=[5, 5, 5]
        # id_common_prefix=['travelling_fire',],
        # ignition_origin=[(3.3, 46.7, 3.82),],
        # ignition_delay_time=[0,],
        # spread_speed=[1,],
        # burning_time=[10,]
    )

    from os.path import basename
    with open(join(dirname(fp_fds_raw), basename(fp_fds_raw.replace('.fds', '.out.fds'))), 'w+') as f:
        f.write(res)


if __name__ == '__main__':
    __test_travelling_fire_line_1_ignition()
    __test_travelling_fire_line_2_ignition()
    __test_travelling_fire_1cw()

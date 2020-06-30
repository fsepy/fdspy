from os.path import join, dirname
from typing import Union, List, Tuple

from fdspy.lib.fds_script_analyser import FDSBaseModel


class FDSTravellingFireMaker(FDSBaseModel):

    def __init__(self, fds_raw: str, id_common_prefix: str, ignition_origin: Union[Tuple, List], spread_speed: float):
        super().__init__(fds_raw=fds_raw)

        # check for columns `CTRL_ID`, `FUNCTION_TYPE`, `RAMP_ID`, `LATCH`, `INPUT_ID` `T` and `F`
        # insert to the dataframe if any one of above does not exist
        fds_df = self.fds_df
        for i in ['CTRL_ID', 'FUNCTION_TYPE', 'RAMP_ID', 'LATCH', 'INPUT_ID', 'T', 'F']:
            if i not in list(fds_df):
                fds_df[i] = None

        # print(self.fds_df[self.fds_df['VENT']])

        for index, row in fds_df[fds_df['_GROUP'] == 'VENT'].iterrows():
            # Make CTRL

            # Make RAMP

            fds_df.iloc[index]['CTRL_ID'] = 0

        print(self.fds_df[self.fds_df['_GROUP'] == 'VENT'])
        pass


if __name__ == '__main__':
    from fdspy import __root_dir__

    with open(join(dirname(__root_dir__), 'tests', 'fds_scripts', 'travelling_fire.fds'), 'r') as f:
        fds_raw = f.read()
    id_common_prefix = 'travelling_fire'
    ignition_origin = (0, 0.5, 0)
    spread_speed = 0.2

    test = FDSTravellingFireMaker(
        fds_raw=fds_raw,
        id_common_prefix=id_common_prefix,
        ignition_origin=ignition_origin,
        spread_speed=spread_speed
    )

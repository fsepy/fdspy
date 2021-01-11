import logging
import re
from abc import ABC

import pandas as pd

logger = logging.getLogger('cli')


class FDSBaseModel(ABC):
    def __init__(self, fds_raw: str):
        self.__fds_raw = None
        self.__fds_df = None

        self.fds_raw = fds_raw
        self.fds_df = self._fds2df(fds_raw)

    def __repr__(self):
        # Instantiate containers for label `l1` and value `l2`
        l1, l2 = list(), list()

        # Make labels and values
        l1.append('CHID'), l2.append(f'{self.fds_df.loc[self.fds_df["CHID"].notnull()]["CHID"][0]}')
        l1.append('No. of lines (exclude blank and comment lines)'), l2.append(f'{len(self.fds_df):d}')
        l1.append('No. of unique head parameters'), l2.append(f'{len(list(set(self.fds_df["_GROUP"]))):d} {list(set(self.fds_df["_GROUP"]))}')
        l1.append('No. of unique parameters (exclude head)'), l2.append(f'{len(self.fds_df.columns)-1:d}')  # -1 to exclude _GROUP which

        # Calculate max length of label and value
        l1_n_char = max(map(len, l1))
        l2_n_char = max(map(len, l2))

        # Make a string consisted of labels and values
        stats = '\n'.join([f'{l1[i]:<{l1_n_char:d}}  {l2[i]:<{l2_n_char:d}.{l2_n_char:d}}' for i in range(len(l1))])

        return stats

    @staticmethod
    def _fds2df(fds_raw: str) -> pd.DataFrame:
        """Converts FDS script to a pandas DataFrame object containing parameterised FDS script for ease processing.

        :param fds_list: FDS script string to be analysed and parameterised.
        :return: a pandas DataFrame object containing parameterised FDS script.
        """

        l0, l1 = FDSBaseModel._fds2list(fds_raw)
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
            fds_group_param_val = FDSBaseModel._fds2list_single_line(i)
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
    def _fds2list_single_line(fds_line: str):
        """Converts a single FDS command in to a list [group_name, parameter1, value1, parameter2, value2, ...]

        :param fds_line: a string contains only one FDS command.
        :return: a list in [group_name, parameter1, value1, parameter2, value2, ...]
        """

        # =======================
        # clean input fds command
        # =======================
        # check command is enclosed within `&` and `/`
        fds_line = re.sub(r"[\n\r]", "", fds_line)  # remove new line characters
        fds_line = fds_line.strip()
        fds_line = re.findall(r"^&(.+)/", fds_line)
        if len(fds_line) == 0:
            # no command found
            return None
        elif len(fds_line) > 1:
            # multiple fds command is supplied, this function only works for single fds command line
            raise ValueError(
                "Multiple lines of command found. "
                "A single line of FDS command should be enclosed within `&` and `/`"
            )
        else:
            # get the command main content only, i.e. this will remove comments or anything outside the `&` and `/`
            fds_line = fds_line[0]

        # ==================
        # Extract group name
        # ==================
        group_name = re.findall(r"^(\w+) ", fds_line)
        if len(group_name) > 1:
            raise ValueError(f"Multiple group names found, only 1 expected: {group_name}")
        elif len(group_name) == 0:
            raise ValueError(f"No group name found. {fds_line}")
        else:
            group_name = group_name[0]

        fds_line = re.sub(r"^\w+ ", "", fds_line)  # remove group_name from the line

        # =============================================================================
        # Rearrange the fds command line into [parameter, value, parameter, value, ...]
        # =============================================================================
        fds_line = re.split(r"(\w+[\(\),\d]* *= *)", fds_line)
        fds_line = list(filter(None, fds_line))
        rep = re.compile(r"[=,]$")
        for i, v in enumerate(fds_line):
            v = v.strip()
            v = rep.sub("", v)
            fds_line[i] = v
        if len(fds_line) % 2 != 0:
            raise ValueError("Attempted to rearrange fds command, not always in `parameter, value` pairs.")

        return [group_name] + fds_line

    @staticmethod
    def _df2fds(fds_df: pd.DataFrame):

        fds_script = list()

        for index, row in fds_df.iterrows():
            line_dict = row.to_dict()
            line_header = line_dict.pop("_GROUP")

            line_content = list()
            for key, val in line_dict.items():
                if val is None:
                    continue
                line_content.append(f"{key}={val}")

            fds_script.append(f"&{line_header} {', '.join(line_content)}/")

        return '\n'.join(fds_script)

    @property
    def fds_raw(self) -> str:
        return self.__fds_raw

    @fds_raw.setter
    def fds_raw(self, v: str):
        self.__fds_raw = v

    @property
    def fds_df(self) -> pd.DataFrame:
        return self.__fds_df

    @fds_df.setter
    def fds_df(self, v: pd.DataFrame):
        self.__fds_df = v


def _test_fds2list_single_line():
    fds2list_single_line = FDSBaseModel._fds2list_single_line

    def len_fds2list_single_line(line_):
        line_ = fds2list_single_line(line_)
        if isinstance(line_, list):
            return len(line_)
        elif line_ is None:
            return None

    line = r"&HEAD CHID='moe1'/"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 3

    line = r"&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 5

    line = r"&MESH ID='stair upper02', IJK=7,15,82, XB=4.2,4.9,-22.0,-20.5,11.1,19.3, MPI_PROCESS=0/"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 9

    line = r"""
    &PART ID='Tracer',
          MASSLESS=.TRUE.,
          MONODISPERSE=.TRUE.,
          AGE=60.0/
    """
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 9

    line = r"&CTRL ID='invert', FUNCTION_TYPE='ALL', LATCH=.FALSE., INITIAL_STATE=.TRUE., INPUT_ID='ventilation'/"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 11

    line = r"&HOLE ID='door - stair_bottom', XB=3.0,3.4,-23.1,-22.3,4.9,6.9/ "
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 5

    line = r"&SLCF QUANTITY='TEMPERATURE', VECTOR=.TRUE., PBX=3.4/"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 7

    line = r"&TAIL /"
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 1

    line = r"""
    &SURF ID='LINING CONCRETE',
          COLOR='GRAY 80',
          BACKING='VOID',
          MATL_ID(1,1)='CONCRETE',
          MATL_MASS_FRACTION(1,1)=1.0,
          THICKNESS(1)=0.2/
    """
    print(fds2list_single_line(line))
    assert len_fds2list_single_line(line) == 13

    line = r"""&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"""
    print(fds2list_single_line(line))
    assert fds2list_single_line(line)[3] == "RESTRICT_TIME_STEP"


def _test_fds2df():
    from fdspy.tests.fds_scripts import general_residential_corridor
    FDSBaseModel(general_residential_corridor)


def _test_df2fds():
    from fdspy.tests.fds_scripts import travelling_fire_1cw
    model = FDSBaseModel(travelling_fire_1cw)
    print(model)
    FDSBaseModel._df2fds(model.fds_df)


if __name__ == '__main__':
    _test_fds2list_single_line()
    _test_fds2df()
    _test_df2fds()

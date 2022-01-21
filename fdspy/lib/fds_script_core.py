import re
from abc import ABC
from typing import Union

import pandas as pd


class GeoOrtho:
    @staticmethod
    def state3d(
            x1: float, y1: float, z1: float,
            x2: float, y2: float, z2: float,
            x3: float, y3: float, z3: float,
            x4: float, y4: float, z4: float
    ) -> int:
        """Check the state between two cuboid (orthogonal) in a 3D space.

        :param x1:
        :param y1:
        :param z1:
        :param x2:
        :param y2:
        :param z2:
        :param x3:
        :param y3:
        :param z3:
        :param x4:
        :param y4:
        :param z4:
        :return:    0 - No contact
                    1 - Contact on single point
                    2 - Contact on edge
                    3 - Contact on face
                    4 - Overlap
        """
        stats_2d = list()
        xyz1 = (x1, y1, z1)
        xyz2 = (x2, y2, z2)
        xyz3 = (x3, y3, z3)
        xyz4 = (x4, y4, z4)
        for plan in ((0, 1), (0, 2), (1, 2)):
            stats_2d.append(
                GeoOrtho.state2d(
                    x1=xyz1[plan[0]], y1=xyz1[plan[1]],
                    x2=xyz2[plan[0]], y2=xyz2[plan[1]],
                    x3=xyz3[plan[0]], y3=xyz3[plan[1]],
                    x4=xyz4[plan[0]], y4=xyz4[plan[1]]
                )
            )

        n_state = [sum(n == i for i in stats_2d) for n in list(range(4))]

        # ====================
        # Check for no contact
        # ====================
        if n_state[0] >= 2:  # if no contact on all three plans
            return 0

        # =======================
        # Check for point contact
        # =======================
        if n_state[1] == 3:  # if point contact on all three plans
            return 1

        # ======================
        # Check for edge contact
        # ======================
        if n_state[1] == 1 and n_state[2] == 2:
            return 2

        # =========================
        # Check for surface contact
        # =========================
        if (n_state[2] == 2 and n_state[3] == 1) or n_state[2] == 3:
            return 3

        # =================
        # Check for overlap
        # =================
        if n_state[3] == 3:
            return 4

        raise ValueError(f'n_state combination not captured, {n_state}, {stats_2d}, {(xyz1, xyz2, xyz3, xyz4)}')

    @staticmethod
    def state2d(
            x1: float, y1: float,
            x2: float, y2: float,
            x3: float, y3: float,
            x4: float, y4: float
    ) -> int:
        """
        For given two rectangles defined by (x1, y1) -> (x2, y2) and (x3, y3) -> (x4, y4), find if they overlap with each
        other.

        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :param x3:
        :param x4:
        :param y3:
        :param y4:
        :return:    0 - separated, no contact
                    1 - contact on point
                    2 - contact on edge
                    3 - overlap on area
        """
        # =================
        # Check for overlap
        # =================
        # identical
        if x1 == x3 and x2 == x4 and y1 == y3 and y2 == y4:
            return 3

        # corner overlap
        xys1 = [(i, j) for i in (x1, x2) for j in (y1, y2)]
        xys2 = [(i, j) for i in (x3, x4) for j in (y3, y4)]
        for (x, y) in xys1:
            if (x3 < x < x4 or x3 > x > x4) and (y3 < y < y4 or y3 > y > y4):
                return 3
        for (x, y) in xys2:
            if (x1 < x < x2 or x1 > x > x2) and (y1 < y < y2 or y1 > y > y2):
                return 3

        # parallel overlap
        if x1 == x3 and x2 == x4:
            if (y1 < y3 < y2) or (y1 > y3 > y2) or (y1 < y4 < y2) or (y1 > y4 > y2):
                return 3
            if (y3 < y1 < y4) or (y3 > y1 > y4) or (y3 < y2 < y4) or (y3 > y2 > y4):
                return 3
        if y1 == y3 and y2 == y4:
            if (x1 < x3 < x2) or (x1 > x3 > x2) or (x1 < x4 < x2) or (x1 > x4 > x2):
                return 3
            if (x3 < x1 < x4) or (x3 > x1 > x4) or (x3 < x2 < x4) or (x3 > x2 > x4):
                return 3

        # cross overlap
        if ((y1 < y3 < y2) or (y1 > y3 > y2)) and ((y1 < y4 < y2) or (y1 > y4 > y2)):
            if (x3 < x1 < x4) or (x3 > x1 > x4) or (x3 < x2 < x4) or (x3 > x2 > x4):
                return 3
        if ((y3 < y1 < y4) or (y3 > y1 > y4)) and ((y3 < y2 < y4) or (y3 > y2 > y4)):
            if (x1 < x3 < x2) or (x1 > x3 > x2) or (x1 < x4 < x2) or (x1 > x4 > x2):
                return 3

        # ========================
        # Check if contact on edge
        # ========================
        for (x, y) in xys1:
            if x3 < x < x4 and (y == y3 or y == y4):
                return 2
            if y3 < y < y4 and (x == x3 or x == x4):
                return 2

        for (x, y) in xys2:
            if x1 < x < x2 and (y == y1 or y == y2):
                return 2
            if y1 < y < y2 and (x == x1 or x == x2):
                return 2

        # =========================
        # Check if contact on point
        # =========================
        n_points = 0
        for a in [(i, j) for i in (x1, x2) for j in (y1, y2)]:
            for b in [(m, n) for m in (x3, x4) for n in (y3, y4)]:
                if a == b:
                    n_points += 1

        if n_points >= 4:
            return 3
        if n_points == 2:
            return 2
        elif n_points == 1:
            return 1

        return 0


class MESH(ABC, GeoOrtho):
    def __init__(self, ID: str, IJK: Union[list, tuple], XB: Union[list, tuple], **kwargs):
        self.id = ID

        if isinstance(IJK, str):
            IJK = [float(i) for i in IJK.split(',')]

        if isinstance(XB, str):
            XB = [float(i) for i in XB.split(',')]

        try:
            self.x1, self.x2, self.y1, self.y2, self.z1, self.z2 = XB
        except Exception as e:
            raise ValueError(f'Error potentially due to XB ({XB}) is not at the right length, {e}')
        try:
            self.i, self.j, self.k = IJK
        except Exception as e:
            raise ValueError(f'Error potentially due to XB ({XB}) is not at the right length, {e}')

        self.misc_kwargs = kwargs

    def to_fds(self):
        # &MESH ID='MESH04', IJK=10,4,10, XB=2.0,3.0,0.2,0.6,0.0,1.0/
        IJK = ','.join([f'{i:g}' for i in [self.i, self.j, self.k]])
        XB = ','.join([f'{i:g}' for i in [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]])

        misc = list()
        for k, v in self.misc_kwargs.items():
            if k == '_GROUP':
                continue
            misc.append(f'{k}={v}')

        if len(misc) > 0:
            return f"&{self.misc_kwargs['_GROUP']} ID={self.id}, IJK={IJK}, XB={XB}, {', '.join(misc)}/"
        else:
            return f"&{self.misc_kwargs['_GROUP']} ID={self.id}, IJK={IJK}, XB={XB}/"

    def __repr__(self):
        return self.id

    def __lt__(self, other) -> bool:
        state = self.state3d(
            self.x1, self.y1, self.z1,
            self.x2, self.y2, self.z2,
            other.x1, other.y1, other.z1,
            other.x2, other.y2, other.z2
        )
        return state == 4 or state == 3

    def __gt__(self, other):
        return self.__lt__(other=other)

    def size_cell(self):
        return self.i * self.j * self.k

    def size_volume(self):
        return abs(self.x1 - self.x2) * abs(self.y1 - self.y2) * abs(self.z1 - self.z2)


class FDSBaseModel(ABC):
    def __init__(self, fds_raw: str = None):
        self.__fds_raw = None
        self.__fds_df = None

        if fds_raw is not None:
            self.fds_raw = fds_raw

    def __repr__(self):
        # Instantiate containers for label `l1` and value `l2`
        l1, l2 = list(), list()

        # Make labels and values
        l1.append('CHID'), l2.append(f'{self.fds_df.loc[self.fds_df["CHID"].notnull()]["CHID"][0]}')
        l1.append('No. of lines (exclude blank and comment lines)'), l2.append(f'{len(self.fds_df):d}')
        l1.append('No. of unique head parameters'), l2.append(
            f'{len(list(set(self.fds_df["_GROUP"]))):d} {list(set(self.fds_df["_GROUP"]))}')
        l1.append('No. of unique parameters (exclude head)'), l2.append(
            f'{len(self.fds_df.columns) - 1:d} {list(self.fds_df.columns)}')  # -1 to exclude _GROUP which

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
        self.__fds_df = self._fds2df(v)

    @property
    def fds_df(self) -> pd.DataFrame:
        return self.__fds_df

    @fds_df.setter
    def fds_df(self, v: pd.DataFrame):
        self.__fds_df = v

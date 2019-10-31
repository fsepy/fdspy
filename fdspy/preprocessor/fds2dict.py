# -*- coding: utf-8 -*-

"""

"""

import re


class FDS2Dict:
    pass


def fds2dict(fds: str):

    # fds = "".join(fds.split())

    rep_0 = re.compile(r'&[\s\S]*?/')
    # rep = re.compile(r'&SURF')

    res = rep_0.findall(fds)

    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

    return res


if __name__ == '__main__':
    from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_ARUP_TUNNEL_FIRE

    out = fds2dict(EXAMPLE_FDS_SCRIPT_ARUP_TUNNEL_FIRE)

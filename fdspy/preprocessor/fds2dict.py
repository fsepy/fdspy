# -*- coding: utf-8 -*-

"""

"""

import re


class FDS2Dict:
    pass


def all_fds_groups_in_a_list(fds_manual_latex: str = None):

    # Parse input, i.e. the manual latex source code
    # ==============================================
    if fds_manual_latex is None:
        from fdspy.preprocessor import FDS_MANUAL_TABLE_GROUP_NAMELIST as _
        out = _
    else:
        out = fds_manual_latex

    # Analyse the source code, extract FDS input parameters
    # =====================================================

    # replace all escaped characters
    out = out.replace("\\", "")
    # remove all commented-out lines
    out = re.sub(r"%[\s\S.]*?[\r|\n]", "", out)
    # remove multiple \n or \r, step 1 - split string
    out = re.split(r"[\r|\n]", out)
    # remove multiple \n or \r, step 2 - remove empty lines
    out = list(filter(None, out))
    # remove multiple \n or \r, step 3 - restore to a single string
    out = '\n'.join(out)
    # find all possible FDS input parameters
    out = re.findall(r"\n{ct\s([\w]*)[(\}]", out)
    # filter out duplicated and sort all the items
    out = sorted(list(set(out)))

    return out


def test_fds_groups_in_a_list():
    assert len(all_fds_groups_in_a_list()) == 36


def all_fds_input_parameters_in_a_list(fds_manual_latex: str = None):
    """Get an exhausted list of input parameters for all groups in Fire Dynamics Simulator.

    :param fds_manual_latex: text string in latex source code obtained from FDS manual source codes.
    :return: a list of all input parameters extracted from the supplied FDS manual latex source code.
    """

    # Parse input, i.e. the manual latex source code
    # ==============================================
    if fds_manual_latex is None:
        from fdspy.preprocessor import FDS_MANUAL_CHAPTER_LIST_OF_INPUT_PARAMETERS as _
        out = _
    else:
        out = fds_manual_latex

    # Analyse the source code, extract FDS input parameters
    # =====================================================

    # replace all escaped characters
    out = out.replace("\\", "")
    # remove all commented-out lines
    out = re.sub(r"%[\s\S.]*?[\r|\n]", "", out)
    # remove multiple \n or \r, step 1 - split string
    out = re.split(r"[\r|\n]", out)
    # remove multiple \n or \r, step 2 - remove empty lines
    out = list(filter(None, out))
    # remove multiple \n or \r, step 3 - restore to a single string
    out = '\n'.join(out)
    # find all possible FDS input parameters
    out = re.findall(r"\n{ct\s([\w]*)[(\}]", out)
    # filter out duplicated and sort all the items
    out = sorted(list(set(out)))

    return out


def test_all_fds_input_parameters_in_a_list():
    assert len(all_fds_input_parameters_in_a_list()) == 652


def fds2dict(fds: str):

    res = re.findall(r'&[\s\S]*?/', fds)

    dict_from_fds = dict()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)
        dict_from_fds[str(i)] = dict()
        dict_from_fds[str(i)]['group'] = re.findall(r'&(\S*)\s', res[i])[0]
        dict_from_fds[str(i)]['parameters'] = re.findall(r'&\S*\s(.+)', res[i])[0]

        # ID = 'FLOW + stair_lobby_door', QUANTITY =

    return dict_from_fds


def fds2dict_input_parameters():
    input = r"&DEVC ID='FLOW + stair_lobby_door', QUANTITY='VOLUME FLOW +', XB=5.1,6.7,-22.1,-22.1,8.5,10.5/"

    res = re.findall(r"[\s|,]+[\w]+=([\S ]+?)[=|/]", input)

    print(res)


if __name__ == '__main__':

    # test_fds_groups_in_a_list()
    # test_all_fds_input_parameters_in_a_list()

    # from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_RIU_MOE1
    # out = fds2dict(EXAMPLE_FDS_SCRIPT_RIU_MOE1)
    #
    # print(out)

    fds2dict_input_parameters()

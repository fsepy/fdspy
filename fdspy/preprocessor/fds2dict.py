# -*- coding: utf-8 -*-

"""

"""

import re


class FDS2Dict:
    pass


def all_fds_input_parameters_in_a_list(fds_manual_latex: str = None):
    """Get an exhausted list of input parameters for all groups in Fire Dynamics Simulator.

    :param fds_manual_latex: text string in latex source code obtained from FDS manual source codes.
    :return: a list of all input parameters extracted from the supplied FDS manual latex source code.
    """

    # Parse input, i.e. the manual latex source code
    # ==============================================
    if fds_manual_latex is None:
        from fdspy.preprocessor import FDS_MANUAL_CHAPTER_LIST_OF_INPUT_PARAMETERS as fds_params
    else:
        fds_params = fds_manual_latex

    # Analyse the source code, extract FDS input parameters
    # =====================================================

    # replace all escaped characters
    fds_params = fds_params.replace("\\", "")
    fds_params = re.split(r"[\r|\n]", fds_params)
    # remove empty strings
    fds_params = list(filter(None, fds_params))
    fds_params = '\n'.join(fds_params)
    fds_params = re.findall(r"\n{ct\s([\w]*)[\(\}]", fds_params)
    # filter out duplicated and sort all the items
    fds_params = sorted(list(set(fds_params)))

    return fds_params


def fds2dict(fds: str):

    res = re.findall(r'&[\s\S]*?/', fds)

    dict_from_fds = dict()
    for i, v in enumerate(res):
        res[i] = re.sub(r"[\n\r]", "", v)

        dict_from_fds['line'] = i
        dict_from_fds['group'] = re.findall(r'&(\S*)\s', res[i])[0]
        dict_from_fds['parameters'] = re.findall(r'&\S*\s(.+)', res[i])[0]

    return res


if __name__ == '__main__':
    # from fdspy.preprocessor import EXAMPLE_FDS_SCRIPT_ARUP_TUNNEL_FIRE
    #
    # out = fds2dict(EXAMPLE_FDS_SCRIPT_ARUP_TUNNEL_FIRE)

    all_fds_input_parameters_in_a_list()

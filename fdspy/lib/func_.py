# coding: utf-8

import itertools
import numpy
import os
import pandas
import copy

import subprocess

# &DEVC ID='THCP', QUANTITY='THERMOCOUPLE', XYZ=5.8,8.6,21.0/


def step1(path_data_directory):

    # FIND ALL .CSV FILES
    # ===================
    dir_cwd = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    path_data_directory = os.path.join(dir_cwd, path_data_directory)
    path_files_list = os.listdir(path_data_directory)

    path_files_csv_list = []
    for path_file in path_files_list:
        if path_file.endswith(".csv"):
            path_files_csv_list.append(os.path.join(path_data_directory, path_file))

    # READ DATA INTO DATAFRAME
    # ========================
    df_data_list = []
    for path_file in path_files_csv_list:
        print(path_file)
        df_data_list.append(
            pandas.read_csv(
                path_file,
                index_col=None,
                error_bad_lines=False,
                header=1,
                skiprows=0,
                dtype=float,
            )
        )

    df_data = pandas.concat(df_data_list, sort=False, ignore_index=True)

    selector_prefix = "VISIBILITY"

    list_headers = df_data.columns.values
    # df_data_ = copy.copy(df_data)
    for k, v in df_data.iteritems():
        print(k)
        if str(k).find(selector_prefix) > -1:
            pass
        else:
            df_data_ = df_data_.drop(k, 0)

    print(df_data_)


def set_bginfo(
    str_desktop_info,
    path_bginfo_text_file,
    path_bginfo_exe,
    path_bginfo_config_file,
    int_bginfo_timer=0,
):
    # str_desktop_info = 'CHI LE MA'
    # path_bginfo_text_file = r"C:\APP\BGInfo\INFO"
    # path_bginfo_exe = r"C:\APP\BGInfo\Bginfo.exe"
    # path_bginfo_config_file = "C:\APP\BGInfo\schedular.bgi"
    # int_bginfo_timer = 0

    with open(path_bginfo_text_file, "w") as f:
        f.write(str_desktop_info)

    sp = subprocess.Popen(
        [
            path_bginfo_exe,
            path_bginfo_config_file,
            r"/timer:{:d}".format(int_bginfo_timer),
        ],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )


if __name__ == "__main__":
    # generate_xyz()
    step1(r"C:\Users\IanFu\Desktop\test_fdspy")

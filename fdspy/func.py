import pandas
import os
from tkinter import Tk, filedialog, StringVar
import itertools
import re
import numpy


def generate_xyz(*list_elements):

    xyz = itertools.product(*list_elements)
    xyz = list(xyz)

    return xyz


def generate_devc(id_prefix, quantity, xyz, is_return_list=False):

    id_fmt = '{quantity}_{xyz}'
    xyz_fmt = '{:.3f},{:.3f},{:.3f}'
    str_fmt = "&DEVC ID='{id}', QUANTITY='{quantity}', XYZ={xyz}/"

    list_cmd = []
    for i in xyz:
        xyz_str = xyz_fmt.format(*i)
        id_str = id_fmt.format(quantity=id_prefix, xyz=xyz_str)
        list_cmd.append(str_fmt.format(id=id_str, quantity=quantity, xyz=xyz_str))

    if is_return_list:
        return list_cmd
    else:
        return '\n'.join(list_cmd)


def read_multi_csv_to_pd(*list_path_csv, header=0, index_col=None, axis=0):
    list_pd_data = []
    for path_csv in list_path_csv:
        list_pd_data.append(pandas.read_csv(
            filepath_or_buffer=path_csv,
            header=header,
            index_col=index_col,
        ))

    return pandas.concat(list_pd_data, axis=1)


def open_files_tk(title='Select Input Files', filetypes=[('csv', ['.csv'])]):
    root = Tk()
    root.withdraw()
    folder_path = StringVar()

    list_paths_raw = filedialog.askopenfiles(title=title, filetypes=filetypes)
    folder_path.set(list_paths_raw)
    root.update()

    list_paths = []
    try:
        for path_input_file in list_paths_raw:
            list_paths.append(os.path.realpath(path_input_file.name))
    except AttributeError:
        return []

    return list_paths

def mtr_calc(
        u_header_prefix='U_VELOCITY',
        v_header_prefix='V_VELOCITY',
        w_header_prefix='W_VELOCITY',
        ske_header_prefix='KSGS',
        path_out=None
):

    path_csv_files = open_files_tk()

    pd_data = read_multi_csv_to_pd(*path_csv_files, header=1, index_col='Time', axis=1)

    list_devc_names = list(pd_data.columns.values)
    devc_name_digits = []

    for devc_name in list_devc_names:
        rep = re.compile('\d+')
        m = re.search(rep, devc_name)
        if m:
            devc_name_digits.append(m.group(0))

    devc_name_digits_set = sorted(list(set(devc_name_digits)))

    dict_out = {}

    for devc_name_digits in devc_name_digits_set:
        label_u = '{}_{}'.format(u_header_prefix, devc_name_digits)
        label_v = '{}_{}'.format(v_header_prefix, devc_name_digits)
        label_w = '{}_{}'.format(w_header_prefix, devc_name_digits)
        label_k = '{}_{}'.format(ske_header_prefix, devc_name_digits)

        u = pd_data[label_u].values
        v = pd_data[label_v].values
        w = pd_data[label_w].values
        k = pd_data[label_k].values

        u_bar = numpy.average(u)
        v_bar = numpy.average(v)
        w_bar = numpy.average(w)

        u_ = (u - u_bar) ** 2
        v_ = (v - v_bar) ** 2
        w_ = (w - w_bar) ** 2

        tke = u_ + v_ + w_
        tke *= 0.5

        mtr = k / (k + tke)

        dict_out['MTR_{}'.format(devc_name_digits)] = mtr

    dict_out['Time'] = pd_data.index

    pd_data_new = pandas.DataFrame.from_dict(dict_out).set_index('Time')

    if path_out is None:
        path_out = filedialog.asksaveasfilename(defaultextension='csv', filetypes=[('csv', ['.csv'])])

    pd_data_new.to_csv(path_out)

    return pd_data_new
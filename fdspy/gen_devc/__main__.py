import itertools
import os
from tkinter import Tk, filedialog, StringVar


def generate_xyz(*list_elements):
    xyz = itertools.product(*list_elements)
    xyz = list(xyz)

    return xyz


def generate_devc(id_prefix, quantity, xyz, is_return_list=False):
    id_fmt = "{quantity}_{index:d}"
    xyz_fmt = "{:.3f},{:.3f},{:.3f}"
    str_fmt = "&DEVC ID='{id}', QUANTITY='{quantity}', XYZ={xyz}/"

    list_cmd = []
    for i, v in enumerate(xyz):
        xyz_str = xyz_fmt.format(*v)

        id_str = id_fmt.format(quantity=id_prefix, index=i)

        list_cmd.append(str_fmt.format(id=id_str, quantity=quantity, xyz=xyz_str))

    if is_return_list:
        return list_cmd
    else:
        return "\n".join(list_cmd)


def open_files_tk(title="Select Input Files", filetypes=[("csv", [".csv"])]):
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


if __name__ == "__main__":
    import numpy as np

    # domain_xyz
    # id_prefix
    # quantity

    # args = sys.argv[1:]

    # generate_

    id_prefix = str(input("ID name prefix: "))

    quantity = str(input("Measurement/Quantity: "))

    x1 = float(input("x lower limit: "))
    x2 = float(input("x upper limit: "))
    x3 = int(input("x devc count: "))
    y1 = float(input("y lower limit: "))
    y2 = float(input("y upper limit: "))
    y3 = int(input("y devc count: "))
    z1 = float(input("z lower limit: "))
    z2 = float(input("z upper limit: "))
    z3 = int(input("z devc count: "))

    if x3 > 1:
        xx = np.linspace(x1, x2, x3)
    else:
        xx = [(x1 + x2) / 2]

    if y3 > 1:
        yy = np.linspace(y1, y2, y3)
    else:
        yy = [(y1 + y2) / 2]

    if z3 > 1:
        zz = np.linspace(z1, z2, z3)
    else:
        zz = [(z1 + z2) / 2]

    xyz = generate_xyz(xx, yy, zz)

    str_devc = generate_devc(
        id_prefix=id_prefix, quantity=quantity, xyz=xyz, is_return_list=False
    )

    path_out = filedialog.asksaveasfilename(
        defaultextension="txt", filetypes=[("txt", [".txt"])]
    )

    with open(path_out, "w") as f:
        f.write(str_devc)

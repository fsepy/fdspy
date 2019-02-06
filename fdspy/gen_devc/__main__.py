from fdspy.func import generate_devc, generate_xyz
from tkinter import filedialog
import sys

if __name__ == '__main__':
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
        xx = [(x1+x2)/2]

    if y3 > 1:
        yy = np.linspace(y1, y2, y3)
    else:
        yy = [(y1+y2)/2]

    if z3 > 1:
        zz = np.linspace(z1, z2, z3)
    else:
        zz = [(z1+z2)/2]

    xyz = generate_xyz(xx, yy, zz)

    str_devc = generate_devc(
        id_prefix=id_prefix,
        quantity=quantity,
        xyz=xyz,
        is_return_list=False
    )

    path_out = filedialog.asksaveasfilename(defaultextension='txt', filetypes=[('txt', ['.txt'])])

    with open(path_out, 'w') as f:
        f.write(str_devc)

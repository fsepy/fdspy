from fdspy.func import mtr_calc
from tkinter import filedialog

if __name__ == "__main__":
    pd_data = mtr_calc(
        u_header_prefix=input("U-Velocity prefix: "),
        v_header_prefix=input("V-Velocity prefix: "),
        w_header_prefix=input("W-Velocity prefix: "),
        ske_header_prefix=input("SKE prefix: "),
        mtr_header_prefix=input("MTR prefix: "),
    )
    path_out = filedialog.asksaveasfilename(
        defaultextension="csv", filetypes=[("csv", [".csv"])]
    )
    pd_data.to_csv(path_out)

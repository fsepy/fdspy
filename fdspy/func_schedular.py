import collections
import json
import time
from fdspy.fds_cls import FdsJob
from tkinter import filedialog, Tk, StringVar
import os


def run_fds(fds_job_config):
    # CREATE FDS JOBS
    # ===============
    j = FdsJob(**fds_job_config)

    # RUN FDS JOBS
    # ============
    j.run_job()

    return j


def select_files():
    # SELECT FDS FILES
    # ================
    list_fds_file = []
    list_dict_fds_job_config = []
    root = Tk()
    root.withdraw()
    path_file = StringVar()
    while True:
        path_fds_file = filedialog.askopenfile(title='Select FDS file', filetypes=[('FDS', ['.fds'])])
        path_file.set(path_fds_file)
        root.update()

        if path_fds_file:
            path_fds_file = os.path.realpath(path_fds_file.name)
            list_fds_file.append(path_fds_file)
            print('Job: {}'.format(path_fds_file))
            list_dict_fds_job_config.append(
                {
                    'path_fds': path_fds_file,
                    'num_mpi': int(input('MPI Process: ')),
                    'num_omp': int(input('OMP Threads: ')),
                }
            )

        else:
            break

    return list_fds_file





if __name__ == '__main__':



    pass

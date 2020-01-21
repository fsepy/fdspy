
import copy

import os
from typing import Union
import collections
import numpy as np
import pandas as pd
import plotly.express as pex
from fdspy.lib.fds_script_proc_decoder import *

# sh_template = '''#!/bin/sh\n\n#SBATCH -J {name}\n#SBATCH -n {n_mpi}\n#SBATCH -p compute\n#SBATCH -e error-%j.err\n#SBATCH -o output-%j.out\n\nsource {fds_source}\nexport OMP_NUM_THREADS={n_omp}\nmpiexec -bootstrap slurm -np $SLURM_NTASKS fds {filename_fds}\n'''
sh_template = '\n'.join([
    '#!/bin/sh',
    '',
    '#SBATCH -J {name}',
    '#SBATCH -n {n_mpi}',
    '#SBATCH -p compute',
    '#SBATCH -e error-%j.err',
    '#SBATCH -o output-%j.out',
    '',
    'source {fds_source}',
    'export OMP_NUM_THREADS={n_omp}',
    '',
    'mpiexec -bootstrap slurm -np $SLURM_NTASKS fds {filename_fds}',
])


def make_sh(
        filepath_fds: str,
        filepath_fds_source: str,
        n_mpi: int = -1,
        n_omp: int = 1,
) -> str:

    # fetch fds script
    with open(filepath_fds, "r") as f:
        fds_script = f.read()

    # parametrise fds script
    # l0, l1 = fds2list(fds_script)
    # d = {i: v for i, v in enumerate(l0)}
    # df = pd.DataFrame.from_dict(d, orient="index", columns=l1)
    df = fds2df(fds_script)

    # work out fds file name
    filename_fds = os.path.basename(filepath_fds)

    # work out job name
    job_name = os.path.basename(filepath_fds).replace('.fds', '')

    # work out number of mpi
    print(n_mpi, type(n_mpi))
    if n_mpi < 1:
        try:
            n_mpi = len(set(df['MPI_PROCESS'].dropna().values))
        except KeyError:
            n_mpi = 1
        n_mpi = 1 if n_mpi < 1 else n_mpi

    # make sh file
    sh = sh_template.format(name=job_name, n_mpi=n_mpi, fds_source=filepath_fds_source, n_omp=n_omp, filename_fds=filename_fds)

    return sh


def test_make_sh():
    filepath_fds = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath_fds = os.path.join(filepath_fds, 'tests_fds', 'residential_corridor.fds')

    sh = make_sh(filepath_fds= filepath_fds)

    print(sh)


if __name__ == '__main__':
    test_make_sh()

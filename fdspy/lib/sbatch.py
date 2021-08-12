import os

sh_template = '\n'.join([
    '#!/bin/sh',
    '',
    '#SBATCH -J {name}',
    '#SBATCH -n {n_mpi}',
    '#SBATCH -p compute',
    '#SBATCH -e error-%j.err',
    '#SBATCH -o output-%j.out',
    '#SBATCH -N 1',
    '#SBATCH --mail-type {mail_type}',
    '#SBATCH --mail-user {mail_user}',
    '',
    'export OMP_NUM_THREADS={n_omp}',
    'export FI_PROVIDER=tcp',
    'export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so',
    '',
    'srun -n $SLURM_NTASKS {filepath_fds_source} {filename_fds}',
])


def make_sh(
        filepath_fds: str,
        filepath_fds_source: str,
        n_mpi: int,
        n_omp: int,
        mail_type: str = 'ALL',
        mail_user: str = 'ian.fu@ofrconsultants.com',
) -> str:

    # fetch fds script
    with open(filepath_fds, "r") as f:
        fds_script = f.read()

    # work out fds file name
    filename_fds = os.path.basename(filepath_fds)

    # work out job name
    job_name = os.path.basename(filepath_fds).replace('.fds', '')

    # make sh file
    sh = sh_template.format(
        name=job_name,
        n_mpi=n_mpi,
        filepath_fds_source=filepath_fds_source,
        n_omp=n_omp,
        filename_fds=filename_fds,
        mail_type=mail_type,
        mail_user=mail_user
    )

    return sh


def test_make_sh():
    filepath_fds = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath_fds = os.path.join(filepath_fds, 'tests_fds', 'residential_corridor.fds')

    sh = make_sh(filepath_fds=filepath_fds)

    print(sh)


if __name__ == '__main__':
    test_make_sh()

from os.path import basename

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
    filename_fds = basename(filepath_fds)

    # work out job name
    job_name = basename(filepath_fds).replace('.fds', '')

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


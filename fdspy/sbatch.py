from os.path import basename

SH_TEMPLATE = (
    '#!/bin/sh\n'
    '\n'
    '#SBATCH -J {name}\n'
    '#SBATCH -n {n_mpi}\n'
    '#SBATCH -p compute\n'
    '#SBATCH -e error-%j.err\n'
    '#SBATCH -o output-%j.out\n'
    '#SBATCH -N 1\n'
    '#SBATCH --mail-type {mail_type}\n'
    '#SBATCH --mail-user {mail_user}\n'
    '\n'
    'export OMP_NUM_THREADS={n_omp}\n'
    'export FI_PROVIDER=tcp\n'
    'export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so\n'
    'source /opt/intel/oneapi/setvars.sh\n'
    'export PATH=$PATH:/opt/intel/oneapi/mpi/latest/lib\n'
    'export PATH=$PATH:/home/installs/FDS679/bin/INTEL/lib\n'
    '\n'
    'srun --mpi=pmi2 -n $SLURM_NTASKS {filepath_fds_source} {filename_fds}'
)


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
    sh = SH_TEMPLATE.format(
        name=job_name,
        n_mpi=n_mpi,
        filepath_fds_source=filepath_fds_source,
        n_omp=n_omp,
        filename_fds=filename_fds,
        mail_type=mail_type,
        mail_user=mail_user
    )

    return sh


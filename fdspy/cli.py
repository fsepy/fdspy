import os
import subprocess

from fdspy import logger
from fdspy.f90nml import read

FP_FDS_SOURCE_TEMPLATE = os.path.join(os.sep+'home', 'installs', 'FDS{}', 'bin', 'fds')


def sbatch(filepath, omp, mpi, fds_version, mail_type, mail_user, *_, **__):
    if mpi == -1:
        try:
            fds_namelist = read(filepath)
            mpi_process_list = list()
            for mesh_namelist in fds_namelist['MESH']:
                if 'MPI_PROCESS' in mesh_namelist:
                    mpi_process_list.append(mesh_namelist['MPI_PROCESS'])
                else:
                    mpi_process_list.append(-1)
            if min(mpi_process_list) > -1:
                mpi = len(set(mpi_process_list))
            elif min(mpi_process_list) == -1 and max(mpi_process_list) == -1:
                mpi = len(fds_namelist['MESH'])
            else:
                mpi = 1
        except Exception as e:
            logger.warning(f'Failed to analyse MPI_PROCESS from *.fds, n_mpi set to 1. {type(e).__name__}.')
            mpi = 1

    # fetch fds source shell script
    try:
        filepath_fds_source = FP_FDS_SOURCE_TEMPLATE.format(f'{int(fds_version)}')
    except ValueError:
        filepath_fds_source = fds_version

    # make sh file
    from fdspy.sbatch import make_sh
    sh = make_sh(
        filepath_fds=filepath,
        filepath_fds_source=filepath_fds_source,
        n_omp=omp,
        n_mpi=mpi,
        mail_type=mail_type,
        mail_user=mail_user
    )

    # write sh file
    filename_sh = 'submit.sh'
    filepath_sh = os.path.join(os.getcwd(), filename_sh)
    with open(filepath_sh, 'w+') as f:
        f.write(sh)

    # run sh file
    subprocess.Popen(['sbatch', filename_sh], cwd=os.getcwd())
    logger.info(f'A job has been submitted: sbatch {filename_sh}')


def helper_get_list_filepath_end_width(cwd: str, end_with: str) -> list:
    filepath_all = list()
    for (dirpath, dirnames, filenames) in os.walk(cwd):
        filepath_all.extend(filenames)
        break  # terminate at level 1, do not go to sub folders.

    filepath_end_with = list()
    for i in filepath_all:
        if i.endswith(end_with):
            filepath_end_with.append(os.path.join(cwd, i))

    return filepath_end_with


def main():
    import argparse

    parser = argparse.ArgumentParser(description='FDS Python Helper Tools')
    subparsers = parser.add_subparsers(dest='sbatch')

    p_sbatch = subparsers.add_parser('sbatch',
        help='To perform `fdspy stats`, generate a `sbatch` shell script file and run the shell script file with '
             '`sbash`.'
    )

    p_sbatch.add_argument('-o', '--omp',
                          help='Number of OMP. Default 1.',
                          nargs='?',
                          default=1,
                          type=int)
    p_sbatch.add_argument('-p', '--mpi',
                          help='Number of MPI. If undefined, -p will be set based on the *.fds file by the following:'
                               '1. If `MPI_PROCESS` is defined for each and every `MESH`: -p will be equal to the greatest MPI_PROCESS.'
                               '2. If `MPI_PROCESS` is defined for some but not all `MESH`: -p will be set to 1.'
                               '3. If `MPI_PROCESS` is not defined for all `MESH`: -p will be set to the number of `MESH`',
                          nargs='?',
                          default=-1,
                          type=int)
    p_sbatch.add_argument('-v', '--fds-version',
                          help=f'FDS source version or full FDS shell script file path. e.g. can be 671 (version number) ',
                          nargs='?',
                          default='671',
                          type=str)
    p_sbatch.add_argument('--mail-type',
                          help='Notify user by email when certain event types occur. Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL (equivalent to BEGIN, END, FAIL, REQUEUE, and STAGE_OUT), STAGE_OUT (burst buffer stage out and teardown completed), TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time limit), TIME_LIMIT_80 (reached 80 percent of time limit), TIME_LIMIT_50 (reached 50 percent of time limit) and ARRAY_TASKS (send emails for each array task). Multiple type values may be specified in a comma separated list. The user to be notified is indicated with --mail-user. Unless the ARRAY_TASKS option is specified, mail notifications on job BEGIN, END and FAIL apply to a job array as a whole rather than generating individual email messages for each task in the job array.',
                          nargs='?',
                          default='NONE',
                          type=str)
    p_sbatch.add_argument('--mail-user',
                          help=f'User to receive email notification of state changes as defined by --mail-type. The default value is the submitting user.',
                          nargs='?',
                          default='NONE',
                          type=str)
    p_sbatch.add_argument('filepath',
                          help=f'Input file name (including extension), use first *.fds file found in the current working directory if not provided.',
                          nargs='?',
                          default=None,
                          type=str)

    args = parser.parse_args()
    import pprint
    pprint.pprint(args)

    # get file path
    if args.filepath is not None:
        args.filepath = os.path.realpath(args.filepath)
    else:
        try:
            args.filepath = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]
        except IndexError:
            raise IndexError(f'Unable to find any *.fds file in directory {os.getcwd()}')

    if args.sbatch:
        sbatch(**args.__dict__)


if __name__ == '__main__':
    main()

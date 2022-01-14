"""fdspy CLI Help.
Usage:
    fdspy pre [<file_name>]
    fdspy post [<file_name>]
    fdspy sbatch [-o=<int>] [-p=<int>] [--mail-type=<str>] [--mail-user=<str>] [-v=<int_str>] [<file_name>]
    fdspy stats [<file_name>]
    fdspy out [--timestats] [<file_name>]

Examples:
    fdspy pre
    fdspy sbatch
    fdspy sbatch -o=2 -p=10 -v=671 --mail-type=END --mail-user=user@email.com example.fds
    fdspy out --timestats example.out

Options:
    -h --help
        To show help.
    -o=<int>
        Optional, 1 by default. To specify number of OMP.
    -p=<int>
        Optional, -1 by default. To specify number of MPI, -1 to be worked out based on `.fds` script.
    -v=<int_str>
        Optional, 671 by default. To specify FDS source version or full FDS shell script file path.
        e.g. can be 671 (version number) or file path /home/installs/FDS671/bin/FDS6VARS.sh.
    --mail-type=<str>
        Optional. Notify user by email when certain event types occur. Valid type values are NONE, BEGIN, END, FAIL,
        REQUEUE, ALL (equivalent to BEGIN, END, FAIL, REQUEUE, and STAGE_OUT), STAGE_OUT (burst buffer stage out and
        teardown completed), TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time limit), TIME_LIMIT_80 (reached 80
        percent of time limit), TIME_LIMIT_50 (reached 50 percent of time limit) and ARRAY_TASKS (send emails for each
        array task). Multiple type values may be specified in a comma separated list. The user to be notified is
        indicated with --mail-user. Unless the ARRAY_TASKS option is specified, mail notifications on job BEGIN, END and
        FAIL apply to a job array as a whole rather than generating individual email messages for each task in the job
        array.
    --mail-user=<str>
        Optional. User to receive email notification of state changes as defined by --mail-type. The default value is
        the submitting user.
    --timestats
        Produce simulation time statistics, saved as `file_name.timestats.csv`.
    <file_name>
        Optional. Input file name (including extension), use first file found in the directory if not provided.

Commands:
    fdspy pre
        Pre-processing. To analyse the `.fds` file in current working directory and produce a summary text file ending
        with `.stats.txt` and a heat release rate curve plot html file ending with `.hrr.html`.
    fdspy post
        Post-processing. To plot heat release rate curve from input `*.fds` and output `*_hrr.csv`.
    fdspy sbatch
        To perform `fdspy stats`, generate a `sbatch` shell script file and run the shell script file with `sbash`.
    fdspy stats
        Will be DEPRECIATED. Identical to `fdspy pre`.
    fdspy out
        Process *.out file
"""

import copy
import os
import subprocess

from docopt import docopt

from fdspy import logger
from fdspy.lib.fds_out_analyser import FDSOutBaseModel
from fdspy.lib.fds_script_analyser import FDSAnalyser

filepath_fds_source_template = '/home/installs/FDS{}/bin/fds'


def stats2(analyser: FDSAnalyser):
    stats_str = ''

    try:
        stats_str = analyser.general()
    except Exception as e:
        logger.error(f'Failed to generate statistics `general`, {e}')
    try:
        stats_str += analyser.mesh()
    except Exception as e:
        logger.error(f'Failed to generate statistics `mesh`, {e}')
    try:
        stats_str += analyser.slcf()
    except Exception as e:
        logger.error(f'Failed to generate statistics `slcf`, {e}')

    if len(stats_str) == 0:
        raise ValueError('len(stats_str)==0 no statistics produced')

    return stats_str


def sbatch(
        filepath_fds: str,
        n_omp: int,
        n_mpi: int,
        fds_v: str,
        mail_type: str,
        mail_user: str
):
    # fetch fds source shell script
    try:
        filepath_fds_source = filepath_fds_source_template.format(f'{int(fds_v)}')
    except ValueError:
        filepath_fds_source = fds_v

    # make sh file
    from fdspy.lib.sbatch import make_sh
    sh = make_sh(
        filepath_fds=filepath_fds,
        filepath_fds_source=filepath_fds_source,
        n_omp=n_omp,
        n_mpi=n_mpi,
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
    arguments = docopt(__doc__)

    logger.debug(f'\n{arguments}')

    # get file path
    if arguments["<file_name>"]:
        file_name = os.path.realpath(arguments["<file_name>"])
        if not os.path.isfile(file_name):
            raise ValueError(f'{file_name} is not a file or does not exits')
    else:
        file_name = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]

    if any([arguments[i] for i in ['pre', 'post', 'sbatch', 'stats']]):
        try:
            with open(file_name, 'r') as f:
                analyser = FDSAnalyser(fds_raw=f.read())
        except Exception as e:
            logger.warning(f'Failed to instantiate FDSAnalyser, {e}')
            analyser = None

    if arguments["stats"] or arguments["pre"] or arguments['sbatch']:
        try:
            stats_string = stats2(analyser)
            with open(file_name + ".stats.txt", "w+", encoding='utf-8') as f:
                f.write(stats_string)
            logger.info('Successfully generated FDS script statistics' + '\n' + stats_string)
        except Exception as e:
            logger.warning(f'Failed to generate FDS script statistics, {e}')

    if arguments["sbatch"]:
        try:
            try:
                df = copy.copy(analyser.fds_df)
                n_mpi = len(set(df['MPI_PROCESS'].dropna().values))
            except Exception as e:
                logger.warning(f'Failed to parse MPI_PROCESS from FDSAnalyser, n_mpi set to 1, {e}')
                n_mpi = 1

            n_omp = arguments['-o'] if arguments['-o'] else 1
            n_mpi = arguments["-p"] if arguments["-p"] else n_mpi
            fds_v = arguments['-v'] if arguments['-v'] else '671'
            mail_type = arguments['--mail-type'] if arguments['--mail-type'] else 'NONE'
            mail_user = arguments['--mail-user'] if arguments['--mail-user'] else 'NONE'

            sbatch(
                filepath_fds=file_name,
                n_omp=int(n_omp),
                n_mpi=int(n_mpi),
                fds_v=fds_v,
                mail_type=mail_type,
                mail_user=mail_user
            )
            logger.info('Successfully produced and executed sbatch')
        except Exception as e:
            logger.warning(f'Failed to execute sbatch, {e}')

    if arguments["post"]:
        logger.warning('post is temporarily removed')

    if arguments["out"]:
        model = FDSOutBaseModel()
        with open(file_name, 'r') as f_:
            fds_out = f_.read()
        model.read_fds_out(fds_out)
        model.make_simulation_time_stats(fp_csv=os.path.realpath(file_name.replace('.out', '') + '.timestats.csv'))

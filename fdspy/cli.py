"""fdspy CLI Help.
Usage:
    fdspy pre [<file_name>]
    fdspy post [<file_name>]
    fdspy sbatch [-o=<int>] [-p=<int>] [--mail-type=<str>] [--mail-user=<str>] [-v=<int_str>] [<file_name>]
    fdspy stats [<file_name>]

Examples:
    fdspy pre
    fdspy sbatch
    fdspy sbatch -o=2 -p=10 -v=671 --mail-type=END --mail-user=user@email.com example.fds

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
"""

import copy
import logging
import os
import subprocess

import pandas as pd
import plotly
import plotly.graph_objects as go
import termplotlib as tpl
from docopt import docopt

from fdspy.lib.fds_script_analyser import ModelAnalyser
from fdspy.lib.fds_script_proc_analyser import main_cli, fds_analyser_hrr
from fdspy.lib.fds_script_proc_decoder import fds2df

c_handler = logging.StreamHandler()
c_handler.setFormatter(
    logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                      '%Y-%m-%d:%H:%M:%S')
)
logger = logging.getLogger('cli')
logger.setLevel(logging.DEBUG)
logger.addHandler(c_handler)
logger.info('fdspy cli started')

filepath_fds_source_template = '/home/installs/FDS{}/bin/FDS6VARS.sh'


def stats2(analyser: ModelAnalyser):
    stats_str = analyser.general()
    stats_str += analyser.mesh()
    stats_str += analyser.slcf()
    stats_str += analyser.hrr_plot()

    return stats_str


def stats(filepath_fds: str):
    dict_out = main_cli(filepath_fds=filepath_fds)

    with open(filepath_fds + ".stats.txt", "w+") as f:
        f.write(dict_out["str"])

    if 'fig_hrr' in dict_out:
        if dict_out['fig_hrr'] is not None:
            try:
                plotly.io.write_html(
                    dict_out["fig_hrr"],
                    file=filepath_fds + ".hrr.html",
                    auto_open=False,
                    config={
                        "scrollZoom": False,
                        "displayModeBar": True,
                        "editable": True,
                        "showLink": False,
                        "displaylogo": False,
                    },
                )
            except Exception as e:
                logger.error(f'Failed to make HRR plot, {e}')

    return dict_out


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
    logger.info('A job has been submitted: ', ' '.join(['sbatch', filename_sh]))


def post(filepath_fds: str):

    # get expected *_hrr.csv file name
    with open(filepath_fds, 'r') as f:
        df = fds2df(f.read())
    chid = df['CHID'].dropna().values[0].replace('"', '').replace("'", '')
    filepath_hrr_csv = os.path.join(os.path.dirname(filepath_fds), f'{chid}_hrr.csv')

    # get time and temperature array from output .csv
    df_hrr_csv = pd.read_csv(filepath_hrr_csv, skiprows=1, index_col=False)
    time_csv = df_hrr_csv['Time'].values
    hrr_csv = df_hrr_csv['HRR'].values

    # get time and temperature array from input .fds
    dict_fds_hrr = fds_analyser_hrr(df)
    time_fds = dict_fds_hrr['time_array']
    hrr_fds = dict_fds_hrr['hrr_array']

    fig = tpl.figure()
    # fig.plot(time_fds, hrr_fds, label="FDS input", width=50, height=15)
    fig.plot(time_fds, hrr_fds, label="FDS output", width=50, height=15)
    fig.show()
    print('asdfsadfsadfsdafasd')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_fds, y=hrr_fds, mode='lines', name='FDS input'))
    fig.add_trace(go.Scatter(x=time_csv, y=hrr_csv, mode='lines', name='FDS output'))
    fig.update_layout(xaxis_title='Time [second]', yaxis_title='HRR [kW]', title=None)
    plotly.io.write_html(
        fig,
        file=f'{filepath_fds}.hrr.html',
        auto_open=False,
        config={
            "scrollZoom": False,
            "displayModeBar": True,
            "editable": True,
            "showLink": False,
            "displaylogo": False,
        },
    )


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
    print(arguments)

    if arguments["<file_name>"]:
        fp_fds_raw = arguments["<file_name>"]
    else:
        fp_fds_raw = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]
    with open(fp_fds_raw, 'r') as f:
        analyser = ModelAnalyser(fds_raw=f.read())

    if arguments["stats"] or arguments["pre"] or arguments['sbatch']:
        try:
            stats_string = stats2(analyser)
            with open(fp_fds_raw + ".stats.txt", "w+", encoding='utf-8') as f:
                f.write(stats_string)
            logger.info('Successfully generated FDS script statistics' + '\n' + stats_string)
        except Exception as e:
            logger.error(f'Failed to generate FDS script statistics, {e}')

    if arguments["sbatch"]:
        try:
            try:
                df = copy.copy(analyser.fds_df)
                n_mpi = len(set(df['MPI_PROCESS'].dropna().values))
            except KeyError:
                n_mpi = 1

            n_omp = arguments['-o'] if arguments['-o'] else 1
            n_mpi = arguments["-p"] if arguments["-p"] else n_mpi
            fds_v = arguments['-v'] if arguments['-v'] else 671
            mail_type = arguments['--mail-type'] if arguments['--mail-type'] else 'NONE'
            mail_user = arguments['--mail-user'] if arguments['--mail-user'] else 'NONE'

            sbatch(
                filepath_fds=fp_fds_raw,
                n_omp=int(n_omp),
                n_mpi=int(n_mpi),
                fds_v=(fds_v),
                mail_type=mail_type,
                mail_user=mail_user
            )
            logger.info('Successfully produced and executed sbatch')
        except Exception as e:
            logger.error(f'Failed to execute sbatch, {e}')

    if arguments["post"]:
        fp_fds_raw = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]
        post(fp_fds_raw)

"""fdspy CLI Help.
Usage:
    fdspy pre
    fdspy post
    fdspy sbatch [-o=<int>] [-p=<int>] [-v=<int_str>]
    fdspy stats

Examples:
    fdspy stats
    fdspy sbatch
    fdspy sbatch -o 2 -p 10 -v 671

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

Commands:
    fdspy pre
        Pre-processing. To analyse the `.fds` file in current working directory and produce a summary text file ending
        with `.stats.txt` and a heat release rate curve plot html file ending with `.hrr.html`.
    fdspy post
        Post-processing. To plot heat release rate curve from input `.fds` and output `*_hrr.csv`.
    fdspy sbatch
        To perform `fdspy stats`, generate a `sbatch` shell script file and run the shell script file with `sbash`.
    fdspy stats
        Will be depreciated. Identical to `fdspy pre`.
"""

import os
import plotly
import subprocess
import pandas as pd
from docopt import docopt
import plotly.graph_objects as go
from fdspy.lib.fds_script_proc_decoder import fds2df
from fdspy.lib.fds_script_proc_analyser import main_cli, fds_analyser_hrr


filepath_fds_source_template = '/home/installs/FDS{}/bin/FDS6VARS.sh'


def stats(filepath_fds: str):

    dict_out = main_cli(filepath_fds=filepath_fds)

    with open(filepath_fds + ".stats.txt", "w+") as f:
        f.write(dict_out["str"])

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

    return dict_out


def sbatch(filepath_fds: str, n_omp: int = 1, n_mpi: int = -1, fds_v: str = 671):

    # fetch fds source shell script
    try:
        filepath_fds_source = filepath_fds_source_template.format(f'{int(fds_v)}')
    except ValueError:
        filepath_fds_source = fds_v

    # make sh file
    from fdspy.lib.sbatch import make_sh
    sh = make_sh(filepath_fds=filepath_fds, filepath_fds_source=filepath_fds_source, n_omp=n_omp, n_mpi=n_mpi)

    # write sh file
    filename_sh = 'submit.sh'
    filepath_sh = os.path.join(os.getcwd(), filename_sh)
    with open(filepath_sh, 'w+') as f:
        f.write(sh)

    # run sh file
    subprocess.Popen(['sbatch', filename_sh], cwd=os.getcwd())
    print('A job has been submitted: ', ' '.join(['sbatch', filename_sh]))


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_fds, y=hrr_fds, mode='lines', name='FDS input'))
    fig.add_trace(go.Scatter(x=time_csv, y=hrr_csv, mode='lines', name='FDS output'))
    fig.update_layout(xaxis_title='Time [minute]', yaxis_title='HRR [kW]')
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

    if arguments["stats"]:
        if arguments["<file_name>"]:
            stats(arguments["<file_name>"])
        else:
            try:
                stats(filepath_fds=helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0])
            except Exception as e:
                # Just print(e) is cleaner and more likely what you want,
                # but if you insist on printing message specifically whenever possible...
                if hasattr(e, "message"):
                    print(e.message)
                else:
                    print(e)

    if arguments["sbatch"]:
        filepath_fds = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]
        stats(filepath_fds=filepath_fds)
        print('FDS script statistics is successfully generated.')
        sbatch(
            filepath_fds=filepath_fds,
            n_omp=arguments["-o"] or 1,
            n_mpi=arguments["-p"] or -1,
            fds_v=arguments["-v"] or 671
        )
        print('Shell file is successfully saved and executed.')

    if arguments["post"]:
        filepath_fds = helper_get_list_filepath_end_width(os.getcwd(), '.fds')[0]
        post(filepath_fds)

"""fdspy CLI Help.
Usage:
    fdspy stats
    fdspy stats <file_name>
    fdspy sbatch [-o=<int>] [-p=<int>] [-v=<int_str>]

Examples:
    fdspy stats
    fdspy stats residential_corridor.fds
    fdspy sbatch
    fdspy sbatch -o 2 -p 10 -v 671

Options:
    -h --help       Help.
    -o=<int>        Optional, 1 by default. To specify number of OMP.
    -p=<int>        Optional, -1 by default. To specify number of MPI, -1 to be worked out based on `.fds` script.
    -v=<int_str>    Optional, 671 by default. To specify FDS source version or full FDS shell script file path. e.g. can
                    be 671 (version number) or file path /home/installs/FDS671/bin/FDS6VARS.sh.


Commands:
    fdspy stats     To analysis all .fds files in the current working directory.
    fdspy sbatch    To perform `fdspy stats`, generate a `sbatch` shell script file and run the shell script file with
                    `sbash`.
"""

import os
import typing
import plotly
import subprocess
from docopt import docopt
from fdspy.lib.fds_script_proc_analyser import main_cli


filepath_fds_source_template = '/home/installs/FDS{}/bin/FDS6VARS.sh'


def stats_single_fds_file(filepath_fds: str):

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


def sbatch_single_fds_file(filepath_fds: str, n_omp: int = 1, n_mpi: int = -1, fds_v: str = 671):

    # fetch fds source shell script
    try:
        filepath_fds_source = filepath_fds_source_template.format(f'{int(fds_v)}')
    except ValueError:
        filepath_fds_source = fds_v

    # if os.path.isfile(os.path.realpath(filepath_fds_source)):
    #     raise ValueError('FDS source shell script not defined.')

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


def main():
    arguments = docopt(__doc__)

    if arguments["stats"]:

        if arguments["<file_name>"]:

            stats_single_fds_file(arguments["<file_name>"])

        else:
            fn = list()
            for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
                fn.extend(filenames)
                break
            for i in fn:
                if i.endswith(".fds"):
                    try:
                        stats_single_fds_file(filepath_fds=i)
                    except Exception as e:
                        # Just print(e) is cleaner and more likely what you want,
                        # but if you insist on printing message specifically whenever possible...
                        if hasattr(e, "message"):
                            print(e.message)
                        else:
                            print(e)

    if arguments["sbatch"]:
        fn = list()
        for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
            fn.extend(filenames)
            break

        for i in fn:
            if i.endswith(".fds"):
                stats_single_fds_file(i)
                print('FDS script statistics is successfully generated.')

                sbatch_single_fds_file(
                    filepath_fds=i,
                    n_omp=arguments["-o"] or 1,
                    n_mpi=arguments["-p"] or -1,
                    fds_v=arguments["-v"] or 671
                )
                print('Shell file is successfully saved and executed.')
                break

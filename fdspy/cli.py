"""fdspy CLI Help.
Usage:
    fdspy stats
    fdspy stats <file_name>
    fdspy sbatch [-o=<int>] [-p=<int>]

Options:
    -h --help       to show help.
    -o=<int>        specify number of OMP. 1 if undefined.
    -p=<int>        specify number of MPI. Will be work out based on `.fds` script if undefined.

Commands:
    fdspy stats     to analysis all .fds files in the current working directory
    fdspy sbatch    to perform `fdspy stats`, generate a `.sh` file and run the `.sh` file with `sbash`.
"""

import os
import plotly
import subprocess
from docopt import docopt
from fdspy.lib.fds_script_proc_analyser import main_cli


def stats_single_fds_file(filepath_fds: str):

    dict_out = main_cli(filepath_fds=filepath_fds)
    print(dict_out["str"])
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


def sbatch_single_fds_file(filepath_fds: str, n_omp: int = 1, n_mpi: int = None):

    # make sh file
    from fdspy.lib.sbatch import make_sh
    sh = make_sh(filepath_fds=filepath_fds, n_omp=n_omp, n_mpi=n_mpi)

    # write sh file
    filename_sh = 'submit.sh'
    filepath_sh = os.path.join(os.getcwd(), filename_sh)
    with open(filepath_sh, 'w+') as f:
        f.write(sh)

    # run sh file
    subprocess.Popen(['sbatch', filename_sh], cwd=os.getcwd())


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
                sbatch_single_fds_file(
                    filepath_fds=i,
                    n_omp=arguments["-o"] or 1,
                    n_mpi=arguments["-p"] or None
                )
                break
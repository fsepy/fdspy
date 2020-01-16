"""fdspy CLI Help.
Usage:
    fdspy stats
    fdspy stats <file_name>
    fdspy sbatch [-o=<int>] [-p=<int>]

Options:
    -h --help       to show help.
    -o              number of OMP, default is 1.
    -p              number of MPI, default will be work out based on `.fds` script.

Commands:
    fdspy stats     to analysis all .fds files in the current working directory
    fdspy sbatch    to perform `fds stats`, generate `.sh` file and run the fds script in `cwd`.
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
        auto_open=True,
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
    print(sh)

    # write sh file
    filename_sh = 'submit.sh'
    print(os.path.join(os.getcwd(), filename_sh))
    with open(os.path.join(os.getcwd(), filename_sh), 'w+') as f:
        f.write(sh)

    # run sh file
    subprocess.call(['sbatch', filename_sh], shell=True)

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
                sbatch_single_fds_file(
                    filepath_fds=i,
                    n_omp=arguments["-o"] or 1,
                    n_mpi=arguments["-p"] or None
                )
                break
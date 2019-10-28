if __name__ == "__main__":

    from fdspy.func import set_bginfo
    import time
    import os
    import json

    os.system("title " + "BGINFO ROUTINE")
    path_fds_queue_file = os.path.join("C:", os.sep, "APP", "fdspy", "fds_queue.json")

    print("bginfo is running...")
    while True:

        path_bginfo_exe = os.path.join("C:", os.sep, "app", "bginfo", "bginfo.exe")
        path_bginfo_config_file = os.path.join(
            "C:", os.sep, "app", "bginfo", "bginfo_config.bgi"
        )
        path_bginfo_config_text = os.path.join(
            "C:", os.sep, "app", "bginfo", "bginfo_text"
        )

        with open(path_fds_queue_file, "r") as f:
            dict_queue = json.load(f)

        table_top_boarder = "{uid}{path_fds}{num_omp}{num_mpi}{progress}{status}\n".format(
            uid="=" * 7,
            path_fds="=" * 20,
            num_omp="=" * 7,
            num_mpi="=" * 7,
            progress="=" * 11,
            status="=" * 10,
        )

        table_header = "{uid:7.7}{path_fds:>20.16}{num_omp:7.7}{num_mpi:7.7}{progress:11.11}{status:10.10}\n".format(
            uid="UID",
            path_fds="Job Name",
            num_omp="OMP",
            num_mpi="MPI",
            progress="Progress",
            status="Status",
        )

        table_content = ""
        table_bottom_boarder = table_top_boarder.replace("\n", "")

        table_content_fmt = "{uid:<7.7}{path_fds:<20.20}{num_omp:<7d}{num_mpi:<7d}{progress:<11.1f}{status:<10d}\n"

        if len(dict_queue) > 0:
            for uid, v in dict_queue.items():
                v["path_fds"] = os.path.basename(v["path_fds"])
                table_content += table_content_fmt.format(uid=uid, **v)
        else:
            table_content = "Empty\n"
        table = table_top_boarder + table_header + table_content + table_bottom_boarder

        with open(path_bginfo_config_text, "w") as f:
            f.write(table)

        set_bginfo(
            path_bginfo_exe=path_bginfo_exe,
            path_bginfo_config_file=path_bginfo_config_file,
        )

        time.sleep(30)

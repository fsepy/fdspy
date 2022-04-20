from fds_script_analyser import *


def test_FDSAnalyser():
    from os import path
    import fdspy

    fp_fds_list = [
        'general-benchmark_1.fds', 'general-error.fds', 'general-residential_corridor.fds', 'general-room_fire.fds',
        'mesh-0.fds', 'mesh-1.fds', 'mesh-2.fds', 'mesh_16_1m.fds', 'travelling_fire-1cw.fds',
        'travelling_fire-line-1_ignition_origin.fds',
    ]
    for fp_fds in fp_fds_list:
        print(fp_fds)

        fp_fds = path.join(fdspy.__root_dir__, 'tests', 'fds_scripts', fp_fds)

        with open(fp_fds, 'r') as f:
            fds_raw = f.read()

        model = FDSAnalyser(fds_raw)

        logger.info(model.hrr_plot(size=(80, 10)))
        model.general()
        model.mesh()
        model.slcf()
        df = copy.copy(model.fds_df)
        try:
            n_mpi = len(set(df['MPI_PROCESS'].dropna().values))
        except:
            n_mpi = '-1'
        print(n_mpi)


if __name__ == '__main__':
    test_FDSAnalyser()

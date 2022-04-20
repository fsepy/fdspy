from fds_out_analyser import *


class Test_FDSOutBaseModel(FDSOutBaseModel):
    def __init__(self):
        super().__init__()
        self.test_make_simulation_time_stats()

    def test_make_simulation_time_stats(self):
        """
        Test function for `FDSOutBaseModel.make_simulation_time_stats`
        """
        from fdspy.tests.fds_out import general_benchmark_1

        self.fds_out = general_benchmark_1
        stats = self.make_simulation_time_stats()  # make simulation time stats
        print(stats)

        # check `FDSOutBaseModel.make_simulation_time_stats` returns the expected dict
        assert all([i in stats for i in ['time_step', 'simulation_time', 'wall_time_elapse']])

        # check no. of time records, should be 29
        assert all([len(stats[i]) == 29 for i in ['time_step', 'simulation_time', 'wall_time_elapse']])


if __name__ == '__main__':
    Test_FDSOutBaseModel()

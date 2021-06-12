import numpy as np

from fdspy.lib.fds_script_core import FDSBaseModel, MESH


class FDSOptimiser(FDSBaseModel):
    def __init__(self, fds_raw: str):
        super().__init__(fds_raw=fds_raw)

    def optimise_mesh(self, ):
        df_fds = self.fds_df

        meshes = list()
        for index, row in df_fds.loc[df_fds['_GROUP'] == 'MESH'].iterrows():
            row.dropna(inplace=True)
            line_dict = row.to_dict()
            meshes.append(MESH(**line_dict))

        weights = np.array([i.size_cell() for i in meshes])
        n_groups = 4

        mesh_optimiser = GroupedCombinations(n_items=len(meshes), n_groups=n_groups, weights=weights)
        best_gcomb = mesh_optimiser.find_best_grouped_combinations()
        best_gcomb_indexes = mesh_optimiser.gcomb_weight2index(weights=weights, gcomb=best_gcomb)

        print(best_gcomb_indexes)
        print(best_gcomb)
        print([sum(i) for i in best_gcomb])
        print(f'VAR:        {np.var([sum(i) for i in best_gcomb])}')


class TestFDSOptimiser(FDSOptimiser):
    def __init__(self, fds_raw: str):
        super().__init__(fds_raw=fds_raw)

    def test_1(self):
        pass

    def test_2(self):
        pass


def _test_mesh_optimiser():
    from fdspy.tests.fds_scripts import mesh_optimiser_0
    model = FDSBaseModel(mesh_optimiser_0)

    df_fds = model.fds_df.copy()

    meshes = list()
    for index, row in df_fds.loc[df_fds['_GROUP'] == 'MESH'].iterrows():
        row.dropna(inplace=True)
        line_dict = row.to_dict()
        meshes.append(MESH(**line_dict))

    edges = list()
    for i, mesh in enumerate(meshes):
        edges.append(list())
        for j, mesh_ in enumerate(meshes):
            edges[-1].append(j)

    import numpy as np
    from fdspy.func_graph import get_gcombs_all, gcombs2best_gcombs
    weights = np.array([i.size_cell() for i in meshes])
    vertices = np.arange(len(meshes))
    n_groups = 4

    gcombs = get_gcombs_all(vertices=vertices, n_groups=n_groups, edges=edges)
    best_gcomb = gcombs2best_gcombs(gcombs=gcombs, weights=weights)[0]

    print(best_gcomb)
    print([[weights[j] for j in i] for i in best_gcomb])
    print([sum([weights[j] for j in i]) for i in best_gcomb])
    print(f'VAR:        {np.var([sum([weights[j] for j in i]) for i in best_gcomb]):g}')

    # for n_mpi, v in enumerate(best_gcomb):
    #     for n_mesh in v:
    #         mesh = meshes[n_mesh]
    #         mesh.misc_kwargs.update({'MPI_PROCESS': f'{n_mpi}'})
    #         print(mesh.to_fds())


def _test_mesh_optimiser_2():
    from fdspy.tests.fds_scripts import mesh_optimiser_2 as mesh_optimiser
    from fdspy.lib.gcomb import GroupedCombinations
    model = FDSBaseModel(mesh_optimiser)

    df_fds = model.fds_df.copy()

    meshes = list()
    for index, row in df_fds.loc[df_fds['_GROUP'] == 'MESH'].iterrows():
        row.dropna(inplace=True)
        line_dict = row.to_dict()
        meshes.append(MESH(**line_dict))

    import numpy as np
    weights = np.array([i.size_cell() for i in meshes])
    n_groups = 12

    mesh_optimiser = GroupedCombinations(n_items=len(meshes), n_groups=n_groups, weights=weights)
    best_gcomb = mesh_optimiser.find_best_grouped_combinations()
    best_gcomb_indexes = mesh_optimiser.gcomb_weight2index(weights=weights, gcomb=best_gcomb)

    print(best_gcomb_indexes)
    print(best_gcomb)
    print([sum(i) for i in best_gcomb])
    print(f'VAR:        {np.var([sum(i) for i in best_gcomb])}')

    for n_mpi, v in enumerate(best_gcomb_indexes):
        for n_mesh in v:
            mesh = meshes[n_mesh]
            mesh.misc_kwargs.update({'MPI_PROCESS': f'{n_mpi}'})
            print(mesh.to_fds())


if __name__ == '__main__':
    _test_mesh_optimiser_2()

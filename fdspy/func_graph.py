from itertools import combinations
from typing import Union

import numpy as np


def groups(vertices, n_groups: int = 1) -> list:
    try:
        assert len(vertices) >= n_groups
    except AssertionError:
        raise ValueError(
            f'Number of vertices ({len(vertices)}) should be greater or equal to number of groups ({n_groups})')

    # work out how many possible arrangements there are
    group_arrangements = [[1, ] * (n_groups - 1) + [len(vertices) - n_groups + 1]]
    while True:
        a = list(group_arrangements[-1])
        is_changed = False

        for i in range(len(a) - 1, 0, -1):
            if a[i] - a[i - 1] > 1:
                a[i - 1] += 1
                a[i] -= 1
                group_arrangements.append(a)
                is_changed = True
                break

        if is_changed is False: break

    group_arrangements = [sorted(i)[::-1] for i in group_arrangements]

    return group_arrangements


def flatten_tuples(list_of_lists: tuple):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], tuple):
        return flatten_tuples(list_of_lists[0]) + flatten_tuples(list_of_lists[1:])
    return list_of_lists[:1] + flatten_tuples(list_of_lists[1:])


def validate_edges(edge_mat: np.ndarray, vertices: np.ndarray):
    if vertices.shape == (1,):
        return True
    for v in vertices:
        if not np.any(edge_mat[v][vertices[vertices != v]]):
            return False
    return True


def _test_validate_edges():
    vertices = np.arange(6)
    edges = [(1, 2, 3), (0, 2, 4), (0, 1), (0, 4), (1, 3, 5), (4,)]

    # construct edge matrix
    edge_mat = np.zeros(shape=(len(vertices), len(vertices)))
    np.fill_diagonal(edge_mat, 1)
    for i, v in enumerate(edges):
        edge_mat[i, v] = 1

    vs_test = [
        [(0,), True],
        [(3,), True],
        [(5,), True],
        [(0, 1), True],
        [(0, 5), False],
        [(1, 4), True],
        [(2, 1), True],
        [(2, 3), False],
        [(2, 5), False],
        [(4, 5), True],
        [(0, 1, 2), True],
        [(2, 4, 5), False],
        [(0, 1, 2, 4), True],
        [(0, 1, 2, 5), False],
        [(0, 1, 2, 3, 4, 5), True],
    ]

    for vertices, answer in vs_test:
        result = None
        try:
            result = validate_edges(edge_mat=edge_mat, vertices=np.array(vertices))
            assert (result == answer)
        except AssertionError:
            raise AssertionError(f'Connectivity of vertices {vertices} is {answer}, but got {result}')


def append_combinations(vertices: np.ndarray, combs: Union[list, tuple], n: int, edge_mat: np.ndarray = None) -> list:
    """
    Input:
        vertices = [1, 2, 3, 4, 5, 6]
        combs = [
            [(1,), (2, 3),],
            [(1,), (2, 4),],
            ...
        ]
        n = 3
    Output:
        combs_new = [
            [(1,), (2, 3), (4, 5, 6)],
            [(1,), (2, 4), (3, 5, 6)],
            ...
        ]

    """

    combs_new = list()
    n_new_valid_comb = 0

    if edge_mat is None:  # do not check whether all vertices are connected
        if len(combs) == 0:
            for comb_new in combinations(vertices, n):
                combs_new.append((comb_new,))
                n_new_valid_comb += 1
        else:
            for comb in combs:
                # items in comb need to be excluded from vertices when making new combinations
                excluded_indexes = flatten_tuples(comb)
                conditions = np.any(np.reshape(vertices, (1, -1)) == np.reshape(excluded_indexes, (-1, 1)), axis=0)

                for comb_new in combinations(vertices[~conditions], n):
                    combs_new.append(comb + (comb_new,))
                    n_new_valid_comb += 1

    else:  # check if all vertices are connected, only return cases that all vertices are connected
        if len(combs) == 0:
            for comb_new in combinations(vertices, n):
                if n == 1 or validate_edges(edge_mat=edge_mat, vertices=np.array(comb_new)):
                    combs_new.append((comb_new,))
                    n_new_valid_comb += 1
        else:
            for comb in combs:
                # items in comb need to be excluded from vertices when making new combinations
                excluded_indexes = flatten_tuples(comb)
                conditions = np.any(np.reshape(vertices, (1, -1)) == np.reshape(excluded_indexes, (-1, 1)), axis=0)

                for comb_new in combinations(vertices[~conditions], n):
                    # is_exist = False
                    # for _ in combs_new:
                    #     if comb_new in _:
                    #         is_exist = True
                    #         break
                    # if is_exist:
                    #     break
                    if n == 1 or validate_edges(edge_mat=edge_mat, vertices=np.array(comb_new)):
                        combs_new.append(comb + (comb_new,))
                        n_new_valid_comb += 1

    if n_new_valid_comb > 0:
        return combs_new
    else:
        return []


def _test_append_combinations():
    vertices = np.arange(6)
    edges = [(1, 2, 3), (0, 2, 4), (0, 1), (0, 4), (1, 3, 5), (4,)]

    edge_mat = np.zeros(shape=(len(vertices), len(vertices)))
    for i, edge in enumerate(edges):
        edge_mat[i, edge] = 1

    a = append_combinations(
        vertices=vertices,
        combs=(
            ((0,), (1, 2),),
            ((0,), (1, 3),)
        ),
        n=3,
    )
    a_ = [((0,), (1, 2), (3, 4, 5)), ((0,), (1, 3), (2, 4, 5))]
    print(a)
    assert a == a_

    b = append_combinations(
        vertices=vertices,
        combs=(
            ((0,), (1, 2),),
            ((0,), (1, 3),)
        ),
        n=3,
        edge_mat=edge_mat
    )
    b_ = [((0,), (1, 2), (3, 4, 5))]
    print(b)
    assert b == b_


def unique_gcombs(gcombs):
    gcombs_ = list()
    for i, g in enumerate(gcombs):
        gcombs_.append(tuple(sorted(g)))
    gcombs_ = sorted(gcombs_)
    gcombs_ = tuple(set(gcombs_))
    return gcombs_


def _test_unique_gcombs():
    a = unique_gcombs(
        (
            ((0,), (1,), (2, 3)),
            ((1,), (0,), (2, 3))
        )
    )
    a_ = (((0,), (1,), (2, 3)),)
    print(a)
    assert a == a_

    b = unique_gcombs(
        (
            ((0,), (1,), (2, 3)),
            ((1,), (0,), (3, 2))
        )
    )
    b_ = (((0,), (1,), (2, 3)), ((0,), (1,), (3, 2)))
    print(b)
    assert b == b_


def get_gcombs(vertices: Union[np.ndarray, tuple, list], group_arrangement: Union[tuple, list],
               edge_mat: np.ndarray = None):
    combs = list()

    for ga in group_arrangement:
        combs = list(append_combinations(vertices=vertices, combs=combs, n=ga, edge_mat=edge_mat))
        combs = unique_gcombs(combs)
        if len(combs) == 0:
            break

    return combs


def _test_gcombs():
    vertices = np.arange(6)
    ga = (1, 1, 1, 1, 2)

    edges = [(1, 2, 3), (0, 2, 4), (0, 1), (0, 4), (1, 3, 5), (4,)]

    edge_mat = np.zeros(shape=(len(vertices), len(vertices)))
    for i, edge in enumerate(edges):
        edge_mat[i, edge] = 1

    a = get_gcombs(vertices=vertices, group_arrangement=ga, edge_mat=edge_mat)
    a = unique_gcombs(a)
    print(len(a))
    print(a)


def get_gcombs_all(vertices: Union[np.ndarray, list, tuple], n_groups: int, edges: Union[tuple, list]):
    assert len(vertices) >= n_groups
    assert len(vertices) == len(edges)

    gas = groups(vertices=vertices, n_groups=n_groups)

    edge_mat = np.zeros(shape=(len(vertices), len(vertices)))
    for i, edge in enumerate(edges):
        edge_mat[i, edge] = 1

    combs = list()
    for ga in gas:
        comb = get_gcombs(vertices=vertices, group_arrangement=ga, edge_mat=edge_mat)
        comb = unique_gcombs(comb)
        combs.extend(comb)

    return combs


def _test_gcombs_all():
    vertices = np.arange(6)
    edges = [(1, 2, 3), (0, 2, 4), (0, 1), (0, 4), (1, 3, 5), (4,)]
    n_groups = 3
    a = get_gcombs_all(vertices=vertices, n_groups=n_groups, edges=edges)
    for i in a:
        print(i)


def _test_gcombs_all_performance():
    n_vertices = 16
    vertices = np.arange(n_vertices)

    edges = [(i - 1, i + 1) for i in vertices]
    edges[0] = (n_vertices - 1, 1)
    edges[-1] = (n_vertices - 2, 0)

    n_groups = n_vertices - 1

    a = get_gcombs_all(vertices=vertices, n_groups=n_groups, edges=edges)

    print(len(a))


def gcombs2gweights(gcombs, weights):
    for i, gcomb in enumerate(gcombs):
        yield [np.sum(weights[list(_)]) for _ in gcomb]


def gweights2gvars(gweights):
    for i in gweights:
        yield np.var(i)


def gcombs2best_gcombs(gcombs, weights):
    variances = np.array(list(gweights2gvars(gcombs2gweights(gcombs=gcombs, weights=weights))))
    aa = np.where(variances == variances.min())[0]
    return [gcombs[i] for i in aa]


def _test_gcombs2gweights():
    vertices = np.arange(6)
    edges = [(1, 2, 3), (0, 2, 4), (0, 1), (0, 4), (1, 3, 5), (4,)]
    weights = np.array((10, 20, 3, 55, 12, 40))
    n_groups = 3
    gcombs = get_gcombs_all(vertices=vertices, n_groups=n_groups, edges=edges)
    gweights = list(gcombs2gweights(gcombs=gcombs, weights=weights))
    gvars = list(gweights2gvars(gweights))
    for i, gcomb in enumerate(gcombs):
        print(gcomb, gweights[i], gvars[i])

    best_gcombs = gcombs2best_gcombs(gcombs=gcombs, weights=weights)
    for i in best_gcombs:
        print('best', i)


def _test_case_1():
    vertices = np.arange(11)
    edges = [
        (2, 5, 8, 6),
        (1, 5, 3, 4),
        (2, 4),
        (3, 2, 5),
        (4, 2, 1, 8, 9, 10, 11),
        (1, 8, 7),
        (6, 8),
        (6, 7, 1, 5, 9),
        (8, 5, 10),
        (9, 5, 11),
        (5, 10),
    ]
    edges = [[j - 1 for j in i] for i in edges]
    print('EDGES:', edges)
    weights = np.array((1, 0.8, 0.4, 0.4, 1.4, 0.8, 0.8, 0.4, 0.75, 0.75, 1.5))
    n_groups = 4

    edge_mat = np.zeros(shape=(len(vertices), len(vertices)))
    for i, edge in enumerate(edges):
        edge_mat[i, edge] = 1
    r = validate_edges(edge_mat=edge_mat, vertices=np.array((2, 3)))
    assert (r is True)

    gcombs = get_gcombs_all(vertices=vertices, n_groups=n_groups, edges=edges)
    gweights = list(gcombs2gweights(gcombs=gcombs, weights=weights))
    gvars = list(gweights2gvars(gweights))
    for i, gcomb in enumerate(gcombs):
        print('GCOMBS, GWEIGHTS, GVARS:', gcomb, gweights[i], f'{gvars[i]:.2f}')

    best_gcombs = gcombs2best_gcombs(gcombs=gcombs, weights=weights)
    for i in best_gcombs:
        print('BEST GCOMB:', i)


if __name__ == '__main__':
    # _test_validate_edges()
    # _test_unique_gcombs()
    # _test_append_combinations()
    # _test_gcombs()
    # _test_gcombs_all()
    # _test_gcombs_all_performance()
    # _test_gcombs2gweights()
    _test_case_1()

"""
===================================
Make customised grouped combination
===================================

------------------
Problem definition
------------------

For a given number of apples, a, and their weights, W. Put them into some sized bags, G, (i.e., each bag can only
accommodate a defined number of apples).

Find the optimal solution so that the variance of G_w is minimal. Where G_w is an array of numbers representing the
weight of each bag.

-------
Example
-------
>>> a = 6
>>> W = [10, 20, 30, 40, 50, 60]
>>> G = [2, 1, 3]
>>> ans = gcomb(a=a, W=W, G=G)
>>> assert ans == [(10, 20, 40), (60, ), (30, 50)]
"""
import numpy as np
from tqdm import tqdm


class GroupedCombinations:
    def __init__(self, n_items: int = None, n_groups: int = None, weights: np.ndarray = None):
        self.__a = n_items  # number of apples
        self.__W = weights  # apple weights
        self.__n = n_groups

    @property
    def a(self):
        return self.__a

    @property
    def W(self):
        return self.__W

    @property
    def n(self):
        return self.__n

    @a.setter
    def a(self, v: int):
        self.__a = v

    @W.setter
    def W(self, v: np.ndarray):
        self.__W = v

    @n.setter
    def n(self, v: int):
        self.__n = v

    def find_best_grouped_combinations(self, print_progress: bool = True):
        groups = self.find_all_ga(self.a, self.n)

        gcomb_best = None
        gcomb_weight_best = None
        gcomb_var_best = None

        for group in tqdm(groups, disable=not print_progress):
            gcomb = self.find_best_gcomb(a=self.a, W=self.W, G=group)
            gcomb_weight = [sum(i) for i in gcomb]
            gcomb_var = np.var(gcomb_weight)
            if gcomb_var == 0:
                return gcomb
            elif gcomb_var_best is None or gcomb_var_best > gcomb_var:
                gcomb_best = gcomb.copy()
                gcomb_weight_best = gcomb_weight
                gcomb_var_best = gcomb_var

        return gcomb_best

    @staticmethod
    def find_all_ga(n_item: int, n_group: int = 1) -> list:
        assert n_item >= n_group

        def flush_right2left(vals: list):
            vals_new = [vals.copy(), ]
            while True:
                is_changed = False
                for i in range(len(vals) - 1, 0, -1):
                    if vals[i] - vals[i - 1] > 1:
                        vals[i - 1] += 1
                        vals[i] -= 1
                        vals_new.append(vals.copy())
                        is_changed = True
                        break
                if is_changed is False: break
            return vals_new

        # work out how many possible arrangements there are
        gas = list()
        for i in range(1, int(n_item / n_group) + 1):
            reminder = n_item - i * (n_group - 1)
            if reminder >= i:
                ga = [i, ] * (n_group - 1) + [reminder, ]
                gas.extend(flush_right2left(ga))
            else:
                break

        return gas

    @staticmethod
    def gcomb_weight2index(weights: np.ndarray, gcomb: list):
        gcomb_indexes = list()
        for i in gcomb:
            gcomb_indexes_ = list()
            for j in i:
                i_min = np.argmin(np.abs(weights - j))
                gcomb_indexes_.append(i_min)
                weights[i_min] = np.inf
            gcomb_indexes.append(gcomb_indexes_)
        return gcomb_indexes

    @staticmethod
    def find_best_gcomb(a: int, W: np.ndarray, G: tuple) -> list:
        assert sum(G) == len(W) == a

        # weights = sorted(W)
        weights = np.sort(W)
        G = sorted(G, reverse=True)

        target = sum(W) / len(G)

        res = list()
        for g in G:
            res_ = list()
            target_ = target / g

            if g == len(weights):
                res_.extend(weights)
            else:
                for g_ in range(g):
                    i_min = np.argmin(np.abs(weights - target_))
                    res_.append(weights[i_min])
                    weights = np.delete(weights, i_min)

            res.append(res_.copy())

        while True:
            a = [sum([j for j in i]) - target for i in res]
            i = np.argmax(a)
            j = np.argmin(a)

            status, l1, l2 = exchange_to_best(res[i], res[j], target=target)
            # print(status)
            if status is True:
                res[i] = l1.copy()
                res[j] = l2.copy()
            else:
                break

        return res

    @staticmethod
    def __flush_right2left(vals: list):
        vals_new = [vals, ]
        while True:
            is_changed = False
            for i in range(len(vals) - 1, 0, -1):
                if vals[i] - vals[i - 1] > 1:
                    vals[i - 1] += 1
                    vals[i] -= 1
                    vals_new.append(vals)
                    is_changed = True
                    break
            if is_changed is False: break
        return vals_new


class TestGroupedCombinations(GroupedCombinations):
    def __init__(self, ):
        super().__init__()

    def run_all(self):
        self.test_groups()
        self.test_find_best_grouped_combinations()

    def test_groups(self):
        res = self.find_all_ga(
            n_item=6,
            n_group=3,
        )
        print(res)

    def test_find_best_grouped_combinations(self):
        self.n = 3
        self.a = 6
        self.W = list(range(1, self.a + 1, 1))
        res = self.find_best_grouped_combinations(print_progress=False)
        print(res)

        self.n = 12
        self.a = 100
        self.W = [i ** 2 for i in list(range(1, self.a + 1, 1))]
        gcomb = self.find_best_grouped_combinations(print_progress=True)
        print(gcomb)
        gcomb_indexes = self.gcomb_weight2index(weights=np.array(self.W, dtype=float), gcomb=gcomb)
        print(gcomb_indexes)


def exchange_to_best(l1, l2, target):
    l1l2 = np.array(l1 + l2)

    comb_mat = np.full(shape=(len(l1) * len(l2) + 1, len(l1) + len(l2)), fill_value=-1, dtype=int)
    comb_mat[:, 0:len(l1)] = 0
    comb_mat[:, len(l1):len(l1) + len(l2)] = 1

    i = 1
    for j in range(0, len(l1)):
        for k in range(len(l1), len(l1) + +len(l2)):
            comb_mat[i, j] += 1
            comb_mat[i, k] -= 1
            i += 1

    def helper_func(v1, v2, t):
        return ((v1 - t) ** 2 + (v2 - t) ** 2) / 2

    comb_mat_var = list()
    for i in comb_mat:
        comb_mat_var.append(helper_func(v1=sum(l1l2[i == 0]), v2=sum(l1l2[i == 1]), t=target))

    i_min = np.argmin(comb_mat_var)
    if i_min == 0:
        return False, l1, l2
    else:
        return True, list(l1l2[comb_mat[i_min, :] == 0]), list(l1l2[comb_mat[i_min, :] == 1])


if __name__ == '__main__':
    test = TestGroupedCombinations()
    test.run_all()

    res = exchange_to_best(
        l1=[1, 2, 3],
        l2=[4, 5, 6],
        target=21 / 2.
    )
    print(res)

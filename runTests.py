from unittest import TestCase
import test
from test import benchBell, mdpTest, mdpTestFed


class Test(TestCase):
    def test_bench_bell(self):
        benchBell()

# runs gridsearch
class QTabularFed(TestCase):
    def test_mdp_test(self):
        convN = 1
        alpha = 1
        fedP = 2
        syncBackups = 1
        # benchBell
        for epsilon in [1]:
            for syncBackups in [10000, 1000, 100, 10, 1]:
                for fedP in [2, 4, 6, 8, 10]:
                    for alpha in [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]:
                        test.epsilon = epsilon
                        test.syncBackups = syncBackups
                        test.fedP = fedP
                        test.alpha = alpha
                        # plt.clf()
                        mdpTest(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))
                        mdpTestFed(fileExt="alpha_" + str(alpha) + "_sync_" + str(syncBackups) + "_eps_" + str(epsilon))

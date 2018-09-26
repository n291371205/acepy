import threading
import random
import numpy as np

class aceThreading:
    """This class implement multi-threading in active learning for multiple 
    random splits experiments.

    It will display the progress of each thead. When all threads reach the 
    end points, it will return k StateIO objects for analysis.

    Once initialized, it can store and recover from any iterations and breakpoints.

    Note that, this class only provides visualization and file IO for threads, but
    not implement any threads. You should construct different threads by your own, 
    and then provide them as parameters for visualization.
    """
    pass


def query(U):
    print(U)
    return U[1]


def main_loop(query_func=object, U=None, L=None):
    Q = query_func(U)
    print(Q)
    L = [L, Q]


if __name__ == '__main__':
    round = 10
    U = np.random.randint(0, 10, size=(10, 10))
    L = np.random.randint(0, 10, size=(10, 10))

    threads = []
    # 创建线程
    for i in range(round):
        t = threading.Thread(target=main_loop, args=(query, U[i, :], L[i, :]))
        threads.append(t)
    # 启动线程
    for i in range(round):
        threads[i].start()
    for i in range(round):
        threads[i].join()

import threading
import random
import numpy as np


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

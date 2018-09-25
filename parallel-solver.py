from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def parentInit():
    # print(rank)
    bounds = [0, 1]
    i = 1
    print(bounds[0], bounds[1])
    while bounds[0] < bounds[1]:

        print(i)
        comm.isend(bounds, dest=i, tag=i % 2)
        req = comm.irecv(source=MPI.ANY_SOURCE)
        bounds = req.wait()
        i += 1
        if i >= size:
            i = 1


def childInit():
    # print('child', rank)
    while True:
        data = comm.recv(source=0)
        print(data)
        data[0] += .01
        data[1] -= .01
        comm.isend(data, dest=0)


if __name__ == "__main__":
    if rank == 0:
        parentInit()
    else:
        childInit()

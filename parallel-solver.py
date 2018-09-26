from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def parentInit():
    bounds = [0, 1]
    childWorking = {}
    for i in range(1, size):
        childWorking[i] = False

    while bounds[0] < bounds[1]:
        allWorking = True
        notWorking = 0
        for j in childWorking:
            if not childWorking[j]:
                allWorking = False
                notWorking = j

        if allWorking:
            data = comm.recv(source=MPI.ANY_SOURCE)
            bounds = data['bounds']
            print(bounds)
            childWorking[data['rank']] = False

        else:
            comm.send(bounds, dest=notWorking)
            childWorking[notWorking] = True


def childInit():
    while True:
        bounds = comm.recv(source=0)
        bounds[0] += .01
        bounds[1] -= .01
        data = {'rank': rank, 'bounds': bounds}
        comm.send(data, dest=0)


if __name__ == "__main__":
    if rank == 0:
        parentInit()
    else:
        childInit()

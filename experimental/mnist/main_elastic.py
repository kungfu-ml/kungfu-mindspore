from kungfu.cmd import launch_multiprocess


def worker(rank):
    from kungfu.python import current_cluster_size, current_rank
    print('rank=%d' % (rank))
    print('kungfu rank: %d, size %d' %
          (current_rank(), current_cluster_size()))


def main():
    np = 3
    launch_multiprocess(worker, np)


main()

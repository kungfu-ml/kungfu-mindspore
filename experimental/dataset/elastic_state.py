import kungfu.python as kf


class Range:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def take(self, n):
        n = min(n, self.b - self.a)
        return Range(self.a, self.a + n)

    def drop(self, n):
        n = min(n, self.b - self.a)
        return Range(self.a + n, self.b)

    def split(self, n):
        return (self.take(n), self.drop(n))

    def partition(self, i, m):
        k = (self.b - self.a) // m
        a1 = self.a + i * k
        b1 = min(a1 + k, self.b)
        return Range(a1, b1)

    def empty(self):
        return self.a >= self.b

    def __str__(self):
        return '[%d, %d)' % (self.a, self.b)

    def begin(self):
        return self.a

    def end(self):
        return self.b

    def len(self):
        return self.b - self.a


class State:
    def __init__(self, a, b):
        self._range = Range(a, b)
        self.need_sync = True

    def _sync(self):
        a = kf.all_reduce_int_max(self._range.a)
        b = kf.all_reduce_int_max(self._range.b)
        self._range = Range(a, b)

    def sync(self):
        if self.need_sync:
            self._sync()
            self.need_sync = False

    def _check_sync(self, f):
        if self.need_sync:
            raise RuntimeError('calling %s when state is not synchonized' % f)

    def global_func(f):
        def g(*args, **kwargs):
            state = args[0]
            state._check_sync(f)
            return f(*args, **kwargs)

        return g

    @global_func
    def finished(self):
        return self._range.empty()

    @global_func
    def resize_cluster(self, new_size):
        self.need_sync, detached = kf.resize(new_size)
        return detached

    @global_func
    def global_next(self, bs):
        cur, self._range = self._range.split(bs)
        return cur

    @global_func
    def local_next(self, bs):
        cur = self.global_next(bs)
        rank = kf.current_rank()
        size = kf.current_cluster_size()
        local = cur.partition(rank, size)
        return local

    @global_func
    def set_bs(self, bs):
        self.bs = bs

    @global_func
    def global_offset(self):
        return self._range.a

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished():
            raise StopIteration
        bs = 100
        local = self.local_next(bs)
        return local

import sys

import mindspore._c_dataengine as cde
import mindspore.dataset.engine as de
from mindspore.dataset import check_mnist_cifar_dataset, replace_none
from mindspore.dataset.engine.datasets import _select_sampler

import inspect

from elastic_sampler import ElasticSampler


def _my_select_sampler(num_samples,
                       input_sampler,
                       shuffle,
                       num_shards,
                       shard_id,
                       non_mappable=False):
    # <mindspore.dataset.engine.samplers.SequentialSampler object at 0x7fe4926b4b50>
    # return _select_sampler(num_samples, input_sampler, shuffle, num_shards,
    #                        shard_id, non_mappable)

    return ElasticSampler(num_samples=num_samples)


class ElasticMnist(de.MappableDataset):
    @check_mnist_cifar_dataset
    def __init__(self,
                 dataset_dir,
                 usage=None,
                 num_samples=None,
                 num_parallel_workers=None,
                 shuffle=None,
                 sampler=None,
                 num_shards=None,
                 shard_id=None,
                 cache=None):

        if num_parallel_workers is None:
            num_parallel_workers = 1
        assert num_parallel_workers == 1
        super().__init__(num_parallel_workers=num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.sampler = _my_select_sampler(num_samples, sampler, shuffle,
                                          num_shards, shard_id)
        print('_select_sampler returns %s' % (self.sampler), file=sys.stderr)
        self.num_samples = num_samples
        self.shuffle_level = shuffle
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.cache = cache
        print('[Python] ElasticMnist created', file=sys.stderr)

    def parse(self, children=None):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('ElasticMnist::parse caller name:', calframe[1][3])  # parse_tree

        if self.cache:
            cc = self.cache.cache_client
        else:
            cc = None

        # <mindspore.dataset.engine.samplers.SequentialSampler object at 0x7fa898f50450>
        print('[Python] !!!! creating cde.KungFuDataNode with sampler: %s' %
              (self.sampler),
              file=sys.stderr)

        node = cde.KungFuDataNode(
            self.dataset_dir,
            self.usage,
            self.sampler,
            cc,
        ).SetNumWorkers(self.num_parallel_workers)
        return node
        # print('[Python] self.num_parallel_workers: %s' %
        #       (self.num_parallel_workers))
        # node.SetNumWorkers(self.num_parallel_workers)
        # return node

    # def get_args(self):
    #     args = super().get_args()
    #     args["dataset_dir"] = self.dataset_dir
    #     args["usage"] = self.usage
    #     args["num_samples"] = self.num_samples
    #     args["shuffle"] = self.shuffle_level
    #     args["sampler"] = self.sampler
    #     args["num_shards"] = self.num_shards
    #     args["shard_id"] = self.shard_id
    #     args[
    #         "cache"] = self.cache.cache_client if self.cache is not None else None
    #     return args

    # def is_shuffled(self):
    #     if self.shuffle_level is None:
    #         return True

    #     return self.shuffle_level or self.sampler.is_shuffled()

    # def is_sharded(self):
    #     if self.num_shards is not None:
    #         return self.num_shards > 1

    #     return self.sampler.is_sharded()


def create_elastic_mnist(data_path, batch_size):
    ds = ElasticMnist(
        dataset_dir=data_path,
        shuffle=False,
    )
    ds = ds.batch(batch_size)
    return ds


class Cursor:
    def __init__(self, length):
        self._length = length

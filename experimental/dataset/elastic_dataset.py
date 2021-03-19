import mindspore.dataset.engine as de
from mindspore.dataset.engine.datasets import _select_sampler
from mindspore.dataset import check_mnist_cifar_dataset, replace_none
import mindspore._c_dataengine as cde


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
        super().__init__(num_parallel_workers=num_parallel_workers)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.sampler = _select_sampler(num_samples, sampler, shuffle,
                                       num_shards, shard_id)
        self.num_samples = num_samples
        self.shuffle_level = shuffle
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.cache = cache

    def parse(self, children=None):
        if self.cache:
            cc = self.cache.cache_client
        else:
            cc = None

        print('!!!! creating KungFuDataNode')
        return cde.KungFuDataNode(
            self.dataset_dir,
            self.usage,
            self.sampler,
            cc,
        ).SetNumWorkers(self.num_parallel_workers)

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
    ds = ElasticMnist(dataset_dir=data_path)
    ds = ds.batch(batch_size)
    return ds


class Cursor:
    def __init__(self, length):
        self._length = length

from mindspore.dataset.engine.samplers import BuiltinSampler
import mindspore._c_dataengine as cde


class ElasticSampler(BuiltinSampler):
    """
    Samples the dataset elements sequentially, same as not having a sampler.

    Args:
        start_index (int, optional): Index to start sampling at. (dafault=None, start at first ID)
        num_samples (int, optional): Number of elements to sample (default=None, all elements).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> dataset_dir = "path/to/imagefolder_directory"
        >>>
        >>> # creates a SequentialSampler
        >>> sampler = ds.SequentialSampler()
        >>> data = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=8, sampler=sampler)
    """
    def __init__(self, start_index=None, num_samples=None):
        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError(
                    "num_samples should be a positive integer "
                    "value, but got num_samples: {}.".format(num_samples))

        if start_index is not None:
            if start_index < 0:
                raise ValueError(
                    "start_index should be a positive integer "
                    "value or 0, but got start_index: {}.".format(start_index))

        self.start_index = start_index
        super().__init__(num_samples)

    def create(self):
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.SequentialSampler(num_samples, start_index)
        c_child_sampler = self.create_child()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def create_for_minddataset(self):
        start_index = self.start_index if self.start_index is not None else 0
        num_samples = self.num_samples if self.num_samples is not None else 0
        c_sampler = cde.MindrecordSequentialSampler(num_samples, start_index)
        c_child_sampler = self.create_child_for_minddataset()
        c_sampler.add_child(c_child_sampler)
        return c_sampler

    def is_shuffled(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_shuffled()

    def is_sharded(self):
        if self.child_sampler is None:
            return False

        return self.child_sampler.is_sharded()

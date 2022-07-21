import os
import json
import redis
import concurrent.futures
import multiprocessing
from math import inf
import pickle
from typing import Optional, Union, Sequence, Iterable, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _worker_init_fn_t, _collate_fn_t


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class AlnairJobDataset(Dataset):
    def __init__(self, keys: List[str] = None):
        """An abstract class subclassing the torch.utils.data.Dataset class
        
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__preprocess__`, supporting pre-processing loaded data. 
        Subclasses should also overwrite meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~AlnairJobDataLoader`.
        
        .. note::
        Subclassing ~AlnairJobDataset will load data under provided keys from Redis to var:`self.__data` as Map<Key, Value>.
        Overwriting meth:`__preprocess__` allows you to replace var:`self.__data` and var:`self.__targets` with
        iteratable variables that can be iterated in meth:`__get_item__`.
        
        Args:
            keys (List, optional): a list of bucket keys. Defaults to None, meaning loading all keys in the bucket.
        """
        
        self.jobname = os.environ.get('JOBNAME')
        self.__keys = keys
        self.qos = self.load_metainfo()['qos']
        self.jobinfo = self.load_jobinfo()
        self.chunks = []
        self.__targets = None
        if self.qos['UseCache']: # use datasource from Redis
            r = dotdict(self.jobinfo.jinfo).ccauth
            r = dotdict(r)
            self.client = redis.RedisCluster(host=r.host, port=r.port, username=r.username, password=r.password)
            self.load_all_redis_keys()
        else:
            import configparser, boto3
            
            parser = configparser.ConfigParser()
            parser.read('/secret/client.conf')
            s3auth = parser['aws_s3']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            self.client = s3_session.client('s3')
            self.bucket_name = self.jobinfo['datasource']['bucket']
            self.load_all_s3_keys()
        self.load_data(0)

    @property
    def data(self):
        return self.__data
    @property
    def keys(self):
        return self.__keys
    @property
    def targets(self):
        return self.__targets
    
    def load(self, key):
        if self.qos['UseCache']:
            val = self.client.get(key)
            if val is None: # cache missing
                with open('/share/cachemiss', 'w') as f:
                    f.writelines(key)
                while True:
                    val = self.client.get(key)
                    if val is not None:
                        break
        else:
            val = self.client.get_object(Bucket=self.bucket_name, Key=key)['Body'].read()
        return val

    def mload(self, chunks):
        if self.qos['UseCache']:
            keys = [x['key'] for x in chunks]
            # with self.client.pipeline() as pl:
            #     for k in keys:
            #         pl.get(k)
            #     vals = pl.execute()
            vals = self.client.mget(keys)
            for i in range(len(vals)):
                if vals[i] is None:
                    with open('/share/cachemiss', 'w') as f:
                        f.writelines(keys[i])
                    while True:
                        val = self.client.get(keys[i])
                        if val is not None:
                            vals[i] = val
                            break
                self.__keys.append(chunks[i]['name'])
                self.__data[chunks[i]['name']] = vals[i]
        else:
            vals = []
            def helper(chunk):
                val = self.client.get_object(Bucket=self.bucket_name, Key=chunk['key'])['Body'].read()
                self.__keys.append(chunk['key'])
                self.__data[chunk['name']] = val
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for chunk in chunks:
                    futures.append(executor.submit(helper, chunk))
            concurrent.futures.wait(futures)
        return vals
                    
    def get_data(self, index):
        if self.qos['LazyLoading']:
            return self.load(self.data[index])
        else:
            return self.data[index]
    
    def get_target(self, index):
        return self.targets[index]
        
    def load_metainfo(self):
        with open('/jobs/{}.json'.format(self.jobname), 'r') as f:
            jobinfo = json.load(f)
        return jobinfo
    
    def load_jobinfo(self):
        if self.qos['UseCache']:
            while True:
                try:
                    with open("/share/{}.json".format(self.jobname), 'rb') as f:
                        jobinfo = json.load(f)
                        break
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    pass
        else:
            jobinfo = self.load_metainfo()
        return dotdict(jobinfo)

    def load_all_redis_keys(self):
        snapshot = dotdict(self.jobinfo.policy).snapshot
        maxmem = self.qos['MaxMemory']*1024*1024
        keys = snapshot if self.__keys is None else self.__keys
        keys = []
        if self.__keys is not None:
            for k in self.__keys:
                keys.append("{%s}%s" % (dotdict(self.jobinfo.jinfo).jobId, k))
        else:
            keys = snapshot
        tmp = []
        for s in self.client.mget(keys):
            for obj in pickle.loads(s).values():
                tmp.extend(obj)
        snapshot = tmp

        if maxmem == 0 or self.qos['LazyLoading']:
            self.chunks.append(snapshot)
        else:
            total_size = 0
            self.chunks.append([])
            for chunk in snapshot:
                s = int(chunk['size'])
                total_size += s
                if total_size > maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(chunk['name']))
                else:
                    self.chunks[-1].append(chunk)
        
    def load_all_s3_keys(self):
        paginator = self.client.get_paginator('list_objects_v2')
        if self.__keys is not None:
            pages = []
            for k in self.__keys:
                pages.extend(paginator.paginate(Bucket=self.bucket_name, Prefix=k))
        else:
            pages = paginator.paginate(Bucket=self.bucket_name)
        
        maxmem = self.qos['MaxMemory']*1024*1024
        maxmem = inf if maxmem==0 else maxmem
        total_size = 0
        self.chunks.append([])
        for page in pages:
            for item in page['Contents']:
                chk = {'name': item['Key'], 'key': item['Key'], 'size': item['Size']}
                s = int(chk['size'])
                total_size += s
                if total_size > maxmem:
                    self.chunks.append([])
                    total_size = 0
                elif s > maxmem:
                    raise Exception('File {} size is greater than assigned MaxMemory.'.format(chk['name']))
                else:
                    self.chunks[-1].append(chk)
    
    def load_data(self, index):
        self.__data = {}
        self.__keys = []        
        if self.qos['LazyLoading']:
            for chunk in self.chunks[index]:
                self.__keys.append(chunk['name'])
                self.__data[chunk['name']] = chunk['key']
        else:
            # def helper(chk):
            #     val = self.load(chk['key'])
            #     self.__keys.append(chk['name'])
            #     self.__data[chk['name']] = val
            # with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            #     futures = []
            #     for chunk in self.chunks[index]:
            #         futures.append(executor.submit(helper, chunk))
            # concurrent.futures.wait(futures)
            self.mload(self.chunks[index])
        self.__data, self.__targets = self.__convert__()
    
    def __convert__(self) -> Tuple[List, List]:
        """convert self.__data

        Return iteratable X, y that can be indexed by __get_item__
        
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError
    
    
class AlnairJobDataLoader(object):
    def __init__(self, dataset: AlnairJobDataset, 
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        r"""
        Data loader. Combines a dataset and a sampler, and provides an iterable over
        the given dataset.

        The :class:`~torch.utils.data.DataLoader` supports both map-style and
        iterable-style datasets with single- or multi-process loading, customizing
        loading order and optional automatic batching (collation) and memory pinning.

        See :py:mod:`torch.utils.data` documentation page for more details.

        Args:
            dataset (AlnairJobDataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
            sampler (Sampler or Iterable, optional): defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
            batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
                returns a batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
                and :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn (callable, optional): If not ``None``, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading. (default: ``None``)
            generator (torch.Generator, optional): If not ``None``, this RNG will be used
                by RandomSampler to generate random self.indexes and multiprocessing to generate
                `base_seed` for workers. (default: ``None``)
            prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
                in advance by each worker. ``2`` means there will be a total of
                2 * num_workers samples prefetched across all workers. (default: ``2``)
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. (default: ``False``)


        .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                    cannot be an unpicklable object, e.g., a lambda function. See
                    :ref:`multiprocessing-best-practices` on more details related
                    to multiprocessing in PyTorch.

        .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                    When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                    it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                    rounding depending on :attr:`drop_last`, regardless of multi-process loading
                    configurations. This represents the best guess PyTorch can make because PyTorch
                    trusts user :attr:`dataset` code in correctly handling multi-process
                    loading to avoid duplicate data.

                    However, if sharding results in multiple workers having incomplete last batches,
                    this estimate can still be inaccurate, because (1) an otherwise complete batch can
                    be broken into multiple ones and (2) more than one batch worth of samples can be
                    dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                    cases in general.

                    See `Dataset Types`_ for more details on these two types of datasets and how
                    :class:`~torch.utils.data.IterableDataset` interacts with
                    `Multi-process data loading`_.

        .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                    :ref:`data-loading-randomness` notes for random seed related questions.
        """
    
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        self.init_loader()
        self.index = 1
        
    def init_loader(self):
        loader = DataLoader(self.dataset, self.batch_size, self.shuffle, self.sampler, self.batch_sampler, self.num_workers, self.collate_fn, 
                            self.pin_memory, self.drop_last, self.timeout, self.worker_init_fn, self.multiprocessing_context, self.generator, 
                            prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
        self.loader = loader._get_iterator()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            data = self.loader.next()
        except StopIteration:
            if self.index == len(self.dataset.chunks):  # epoch is down
                self.index = 1
                raise StopIteration
            else:
                self.dataset.load_data(self.index)
                self.init_loader()
                data = self.loader.next()
                self.index += 1
        return data
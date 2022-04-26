import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class PipelineDataset(Dataset):
    def __init__(self, dataset, process, params):
        self.dataset = dataset
        self.process = process
        self.params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        processed = self.process(item, **self.params)
        return processed


class PipelineIterator(IterableDataset):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:            
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size        
        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        if isinstance(self._loader_batch_data, torch.Tensor):            
            result = self._loader_batch_data[self._loader_batch_index]
        else:            
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if k in {"hidden_states", "past_key_values", "attentions"} and isinstance(element, tuple):                   
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    continue
                if isinstance(element[self._loader_batch_index], torch.Tensor):                    

                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):                    
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:                    
                    loader_batched[k] = element[self._loader_batch_index]            
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:            
            return self.loader_batch_item()

        
        item = next(self.iterator)
        processed = self.infer(item, **self.params)        
        if self.loader_batch_size is not None:            
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:                
                self.loader_batch_size = observed_batch_size            
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:            
            return processed


class PipelineChunkIterator(PipelineIterator):
    def __init__(self, loader, infer, params, loader_batch_size=None):        
        
        super().__init__(loader, infer, params)

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:            
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            # Try to return next item
            processed = next(self.subiterator)
        except StopIteration:
            # When a preprocess iterator ends, we can start lookig at the next item
            # ChunkIterator will keep feeding until ALL elements of iterator
            # all have created their subiterator and have been iterating against.
            #
            # Another way to look at it, is we're basically flattening lists of lists
            # into a single list, but with generators
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


class PipelinePackIterator(PipelineIterator):
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            processed = self.infer(next(self.iterator), **self.params)
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:                    
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    item = self.loader_batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop("is_last")
                accumulator.append(item)
        return accumulator


class KeyDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]

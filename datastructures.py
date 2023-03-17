from collections import deque
import numpy as np
from torch.utils.data.dataset import IterableDataset
import torch
from core import stack_dicts_list

# returns stacked dict of experiences
class DictBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size) -> dict:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        lines = [self.buffer[idx] for idx in indices]
        output = stack_dicts_list(lines)

        return output

# returns lines of tuples
class LineBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size) -> list:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # original states, actions, rewards, dones, next_states
        lines = [self.buffer[idx] for idx in indices]

        return lines


class LineDataset(IterableDataset):
    def __init__(self, buffer: LineBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        lines = self.buffer.sample(self.sample_size)
        for i in range(len(lines)):
            yield lines[i]

# batch is a batch list of lines of Model x Batch x Data shaped tensors
# turn into a single line with batched items
def VectorCollator(batch):
    batch = list(zip(*batch))
    batch = [torch.cat(x, dim=1) for x in batch]
    return batch

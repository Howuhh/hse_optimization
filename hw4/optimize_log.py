import numpy as np

from typing import List
from dataclasses import dataclass
from dataclasses import field


@dataclass
class OptimizeLog:
    start_time: float
    time: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    grad_info: List[float] = field(default_factory=list)
    call_count: List[int] = field(default_factory=list)
    
    def add_log(self, time, entropy, grad_info, call_count):
        self.time.append(time - self.start_time)
        self.entropy.append(entropy)
        self.grad_info.append(grad_info)
        self.call_count.append(call_count)
        
    def get_log(self):
        logs = [len(self.time), self.time, self.entropy, self.grad_info, self.call_count]
        names = ["num_iter", "time", "entropy", "grad_info", "oracle_calls"]
        
        return {name: np.array(arr) for name, arr in zip(names, logs)}
    
    def get_log_last(self):
        return self.entropy[-1], len(self.time), self.call_count[-1], self.time[-1], self.grad_info[-1]
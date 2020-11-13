import numpy as np

from typing import List
from dataclasses import dataclass
from dataclasses import field


@dataclass
class OptimizeLog:
    start_time: float
    time: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)
    grad_info: List[float] = field(default_factory=list)
    call_count: List[int] = field(default_factory=list)
    
    def add_log(self, time, entropy, alpha, grad_info, call_count):
        self.time.append(time - self.start_time)
        self.entropy.append(entropy)
        self.alpha.append(alpha)
        self.grad_info.append(grad_info)
        self.call_count.append(call_count)
        
    def get_log(self):
        logs = [self.time, self.entropy, self.alpha, self.grad_info, self.call_count]
        names = ["time", "entropy", "lr", "grad info", "call count"]
        
        return {name: np.array(arr) for name, arr in zip(names, logs)}
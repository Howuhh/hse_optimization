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
    
    def add_log(self, time, entropy, alpha, grad_info):
        self.time.append(time - self.start_time)
        self.entropy.append(entropy)
        self.alpha.append(alpha)
        self.grad_info.append(grad_info)
        
    def get_log(self):
        logs = [self.time, self.entropy, self.alpha, self.grad_info]
        
        return [np.array(arr) for arr in logs]
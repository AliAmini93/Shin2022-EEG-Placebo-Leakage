"""Frozen EEGNet-8,2 architecture and conservative GPU-occupancy guard.

PyTorch imported only when the GPU runner is invoked. Do not reuse this
model as a novel contribution: it matches the frozen benchmark schedule.
"""
import os
import subprocess
import time
from pathlib import Path
import torch
from torch import nn

def foreign_gpu_processes(min_mib=300):
    """PIDs of other processes holding GPU memory (owner jobs). We never preempt them: we wait."""
    try:
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ['nvidia-smi-unavailable']
    out = []
    for line in q.strip().splitlines():
        try:
            pid, mem = [int(x) for x in line.split(',')]
        except ValueError:
            continue
        if pid != os.getpid() and mem >= min_mib :
            out.append(pid)
    return out


def wait_for_free_gpu(log, poll=60, need_free_polls=5):
    free = 0
    waited = 0
    while free < need_free_polls:
        busy = foreign_gpu_processes()
        if busy:
            free = 0
            if waited % 900 == 0:
                print(f'GPU busy with owner process(es) {busy}; waiting (no preemption) waited={waited}s', flush=True)
        else:
            free += 1
        if free < need_free_polls:
            time.sleep(poll)
            waited += poll
    return waited



class EEGNet(nn.Module):
    def __init__(self, C=16, T=250, F1=8, D=2, F2=16, kern=125, p=0.25):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1))
        self.dw = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False)
        self.b2 = nn.Sequential(nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b3 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            n = self._feat(torch.zeros(1, 1, C, T)).shape[1]
        self.fc = nn.Linear(n, 2)

    def _feat(self, x):
        return self.b3(self.b2(self.dw(self.b1(x)))).flatten(1)

    def forward(self, x):
        return self.fc(self._feat(x))

    def max_norm(self):
        with torch.no_grad():
            self.dw.weight.data = torch.renorm(self.dw.weight.data, 2, 0, 1.0)
            self.fc.weight.data = torch.renorm(self.fc.weight.data, 2, 0, 0.25)



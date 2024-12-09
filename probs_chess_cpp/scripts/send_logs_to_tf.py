from collections import Counter
import sh
import sys
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummaryWriter(SummaryWriter):
    """
    Wrapper around tensorboard to add points one by one
    """
    def __init__(self):
        super().__init__()
        self.points_cnt = Counter()
        self.figure_cnt = Counter()

    def append_scalar(self, name, value):
        step = self.points_cnt[name]
        self.points_cnt[name] += 1
        self.add_scalar(name, value, step)


def monitor_logs(logfilename: str):
    tensorboard = TensorboardSummaryWriter()

    def __on_new_line(line):
        line = line.rstrip()
        print(line)
        if ': ' in line:
            scalar_name, scalar_val = line.split(': ')
            scalar_val = float(scalar_val)
            tensorboard.append_scalar(scalar_name, scalar_val)    

    with open(logfilename) as file:
        for line in file:
            __on_new_line(line)

    for line in sh.tail("-f", logfilename, _iter=True):
        print(line)
        __on_new_line(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python send_logs_to_tf.py <path_to_log_file>")
    
    else:
        monitor_logs(sys.argv[-1])

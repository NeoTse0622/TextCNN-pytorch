import torch

class Config:
    def __init__(self):
        self.vocab_size = 30000  # 根据实际情况调整
        self.embed_size = 300
        self.num_classes = 10  # 根据实际情况调整
        self.kernel_sizes = [3, 4, 5]
        self.num_channels = 100
        self.dropout = 0.5
        self.batch_size = 64
        self.lr = 1e-3
        self.num_epochs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 8  # 线程数，根据需要调整

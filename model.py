import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes=[3, 4, 5], num_channels=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_channels, (k, embed_size)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

    def forward(self, x):
        '''5*7句子模矩阵'''
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_size)
        '''卷积(3种卷积核尺寸为2，3，4，每种卷积核数量为2，共计6个)  长为5高为4、3、2'''
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # list of (batch_size, num_channels, seq_len)
        '''激活函数'''
        '''每种卷积对应两个特征向量 长为1高为4、5、5'''
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # list of (batch_size, num_channels)
        '''通过最大池化层分别提取2个更高级别的特征，再将其串联起来(三个1*2串联起来)'''
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes) * num_channels)
        x = self.dropout(x)
        '''将卷积池化得到的特征向量通过全连接层映射到标签域，并通过Softmax层得到文本属于每一类的概率取概率最大的类作为文本的标签'''
        x = self.fc(x)  # (batch_size, num_classes)
        return x

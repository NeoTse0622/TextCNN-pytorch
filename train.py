import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import TextCNN
from dataloader import TextDataset
from config import Config
from tqdm import tqdm
import os
import time


config = Config()


'''# 创建目录
    mkdir bert-base-chinese
    
    # 下载 config.json
    curl -o "bert-base-chinese/config.json" https://huggingface.co/bert-base-chinese/resolve/main/config.json
    
    # 下载 vocab.txt
    curl -o "bert-base-chinese/vocab.txt" https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt
    
    # 下载 pytorch_model.bin
    curl -o "bert-base-chinese/pytorch_model.bin" https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin

'''
try:
    print("尝试加载本地BertTokenizer...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    print("本地BertTokenizer加载成功！")
except Exception as e:
    print(f"无法加载本地tokenizer，请检查目录路径。错误信息：{e}")
    exit()

def load_data(file_path):
    dataset = TextDataset(file_path, tokenizer)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

print("###加载训练集ing...###")
train_loader = load_data('./dataset/data_train.csv')
print("###加载验证集ing...###")
val_loader = load_data('./dataset/data_validation.csv')

# 初始化模型
model = TextCNN(config.vocab_size, config.embed_size, config.num_classes, config.kernel_sizes, config.num_channels,
                config.dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# 训练模型
train_losses = []
val_losses = []
val_accuracies = []

def evaluate_accuracy(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print("开始训练模型...")

for epoch in range(config.num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))

    print(
        f"Epoch [{epoch + 1}/{config.num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    with open('./model_data/epoch_loss.txt', 'a') as f:
        f.write(
            f"Epoch {epoch + 1}, Start Time: {epoch_start_time}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}\n")

    # 每隔2个epoch评估一次验证集的精度
    if (epoch + 1) % 2 == 0:
        val_accuracy = evaluate_accuracy(val_loader, model, device)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Val Accuracy: {val_accuracy:.2f}%")

# 绘制loss图
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./model_data/loss_plot.png')
plt.show()

# 绘制精度图
if val_accuracies:
    plt.figure(figsize=(10,5))
    plt.title("Validation Accuracy")
    plt.plot(range(2, config.num_epochs+1, 2), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig('./map_out/accuracy_plot.png')
    plt.show()

# 保存模型
torch.save(model.state_dict(), "textcnn_model.pth")

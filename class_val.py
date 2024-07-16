import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from config import Config
from model import TextCNN
from dataloader import TextDataset
from matplotlib.font_manager import FontProperties

config = Config()
# 自定义配置或硬编码
val_file = './dataset/data_validation.csv'  # 验证集文件路径
# 设置已训练好的模型权重
model_path='textcnn_model.pth'

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')


# 标签映射字典
label_map = {
    0: '体育',
    1: '财经',
    2: '房产',
    3: '家居',
    4: '教育',
    5: '科技',
    6: '时尚',
    7: '时政',
    8: '游戏',
    9: '娱乐'
}

def class_val():
    # 加载验证集数据集
    val_dataset = TextDataset(val_file, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)  # 可根据需要调整 batch_size 和 num_workers

    # 初始化模型
    model = TextCNN(config.vocab_size, config.embed_size, config.num_classes, config.kernel_sizes, config.num_channels, config.dropout)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    model.eval()

    # 验证模型
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['input_ids'], batch['labels']
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算每个类别的准确率
    accuracy_per_class = {}

    for label_id, label_name in label_map.items():
        indices = [i for i, true_label in enumerate(true_labels) if true_label == label_id]
        if len(indices) > 0:
            true_label_sublist = [true_labels[i] for i in indices]
            predicted_label_sublist = [predicted_labels[i] for i in indices]
            accuracy = accuracy_score(true_label_sublist, predicted_label_sublist)
            accuracy_per_class[label_name] = accuracy

    # 设置中文字体
    font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_per_class.keys(), accuracy_per_class.values())
    plt.xlabel('新闻类别', fontproperties=font)
    plt.ylabel('准确率', fontproperties=font)
    plt.title('验证集每个新闻类别的准确率', fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.tight_layout()
    plt.savefig(os.path.join('./model_data', 'class_val_accuracy.png'))
    plt.show()

if __name__ == '__main__':
    class_val()

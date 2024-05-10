import os
import torch
import time
import torch.nn as nn
import pandas as pd
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

a = '*'
#网络参数
input_size = 5
hidden_size = 64
output_size = 3
num_layers = 2
learning_rate = 0.001
batchsize = 128
num_epochs = 200
train_dir='train2'
test_dir='test2' 
writer = SummaryWriter('logs')

def create_sequences(data, sequence_length):
    X, y = [], []
    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    data=data.values  
    data=mm_x.fit_transform(data)
    data_x=data[:,:5]
    data_y=data[:,:3]
    for i in range(len(data) - sequence_length-1):
        X.append(data_x[i:(i+sequence_length),:])
        y.append(data_y[i+sequence_length,:])
        
    X,y=np.array(X),np.array(y)
    return X, y

class AircraftDataset(Dataset):
    def __init__(self, data_folder, sequence_length,train):
        self.data = []
        self.scaler = MinMaxScaler()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        total_files = len(os.listdir(data_folder))
        sample_count = 0

        for idx, file in enumerate(os.listdir(data_folder)):
            data = pd.read_csv(os.path.join(data_folder, file))
            if not data.empty:
                
                X, y = create_sequences(data, sequence_length)
                # print(X)
                # print(y)
                X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
                y = [torch.tensor(seq, dtype=torch.float32)for seq in y]
               
                
                self.data.extend([(X[i], y[i]) for i in range(len(X))])
                
                sample_count += 1
                completion = sample_count / total_files * 100
                
                progress_bar = ' {:.2f}% [{}{}]'.format(completion, '*' * int(completion), '.' * (100 - int(completion)))
 
                print('\r' + progress_bar, end='')
            else:
                print('Empty data in file:', file)
            break
     
        if(train):
            print('\n'+'----------------------Train Dataset Loaded Sucessfully!----------------------')
        else:
            print('\n'+'----------------------Test Dataset Loaded Sucessfully!----------------------')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = AircraftDataset(data_folder=train_dir, sequence_length=10,train=True)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,pin_memory=True)

test_dataset = AircraftDataset(data_folder=test_dir, sequence_length=10,train=False)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True,pin_memory=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out
    
#创建模型实例，简历优化器和损失函数，在GPU上并行计算
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print('----------------------Start Training...----------------------')

batch_idx = 0
for epoch in range(num_epochs):
    start_time = time.time()
    batch_idx += 1
    model.train()
    running_loss = 0
    running_loss_test = 0
    for (X_batch, y_batch),(X_test_batch,y_test_batch) in zip(train_loader,test_loader):
        X_batch = X_batch.to(device)
        X_test_batch = X_test_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        # print(y_batch)
        # print(outputs)
        outputs = outputs.to('cuda:0')
        y_batch = y_batch.to('cuda:0')
        loss = criterion(outputs, y_batch)
        outputs_test = model(X_test_batch)
        outputs_test = outputs.to('cuda:0')
        y_test_batch = y_batch.to('cuda:0')
        loss_test = criterion(outputs_test, y_test_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss_test += loss_test.item()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch {epoch + 1} 耗时： {elapsed_time} 秒")
    average_loss = running_loss / len(train_loader)
    average_loss_test = running_loss_test / len(test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}],Train Average Loss: {average_loss:.16f},Test Average Loss: {average_loss_test:.16f}')
    writer.add_scalar('Loss/train', average_loss, epoch)
    # 在每个epoch结束后输出预测结果与真实结果
    # model.eval() 
    # with torch.no_grad():
    #     for X_batch, y_batch in train_loader:
    #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         # print(X_batch,y_batch)
    #         outputs = model(X_batch)
    #         for i in range(len(outputs)):
    #             predicted = outputs[i]
    #             target = y_batch[i]
    #             print(f'Predicted: {predicted.cpu().numpy()}, Target: {target.cpu().numpy()}')
    
filename = f'MODEL/TAL{average_loss_test:.4f}_MB{batchsize}_E{num_epochs}_LR{learning_rate}.pth'
torch.save(model.state_dict(), filename)
writer.close()
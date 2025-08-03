import torch.nn as nn
import torch
import matplotlib as plt
# Generate Data

import pickle
import numpy as np
data = []
for i in range(1001):
    new_file_num = str(i)
    while len(new_file_num) < 4:
        new_file_num = "0" + new_file_num
    with open(f"/Users/davidzoro/Downloads/RDE1D-main/_output/fort.q{new_file_num}", "rb") as f:
        for _ in range(5):
            next(f) 
        temp_data = np.loadtxt(f)
    data.append(temp_data)
data = np.vstack(data)

labels = data[:, 3]
inputs = data[:,:3]

rows = data.shape[0]

train_inputs = inputs[:int(.9*(rows)), :]
train_labels = labels[:int(.9*(rows))]

validation_inputs = inputs[int(.9*(rows)):, :]
validation_labels = labels[int(.9*(rows)):]

# Model
class UNET (nn.Module):
    def __init__(self, in_channels):
        super(UNET, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels= in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=in_channels // 2,out_channels=  in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels // 2, out_channels= in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose1d(in_channels=in_channels // 8,out_channels=  in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels // 8, out_channels= in_channels // 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=in_channels // 16, out_channels= in_channels // 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=in_channels // 32, out_channels= in_channels // 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose1d(in_channels=in_channels // 64,out_channels=  in_channels // 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(in_channels=in_channels // 64,out_channels=  in_channels // 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(in_channels=in_channels // 128,out_channels=  in_channels // 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(in_channels=in_channels // 256, out_channels= in_channels // 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels, d_model, num_heads, num_encoder_layers, num_classes):
        super(Transformer, self).__init__()
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_encoder_layers)
        self.regressor = nn.Linear(3, num_classes)
        self.the_UNET = UNET(in_channels)
    def forward(self, x):
        x = self.TransformerEncoder(x)
        x = x.mean(dim=0)
        x = self.the_UNET(x)
        x = self.regressor(x)
        return x


#Define hyperparameters
epochs = 50
loss_func = nn.MSELoss()
Model = Transformer(in_channels=512, d_model=3, num_heads=1, num_encoder_layers=1, num_classes=1)
learning_rate = 0.001
optimizer = torch.optim.Adam(Model.parameters(), lr = learning_rate)
# Identifying tracked values

train_loss = []


# training loop
train_inputs = torch.from_numpy(train_inputs).float()
train_labels = torch.from_numpy(train_labels).float()
validation_inputs = torch.from_numpy(validation_inputs).float()
validation_labels = torch.from_numpy(validation_labels).float()

for i in range(epochs):
    optimizer.zero_grad()
    predictions = Model(train_inputs)
    loss = loss_func(predictions, train_labels)
    print(f"Epoch {i} Loss: {loss}")
    train_loss.append(loss)
    loss.backward()
    optimizer.step()


plt.plot(train_loss)
plt.xlabel("Epochs")
plt.ylabel("Mean-Squared Error")

with torch.no_grad():
    val_predictions = Model(validation_inputs)
    loss = loss_func(val_predictions, validation_labels)

print("Validation Loss:", loss)

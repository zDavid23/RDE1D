import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from vpython import *

# Generate Data

import pickle
import numpy as np
data = []
for i in range(1001):
    new_file_num = str(i)
    while len(new_file_num) < 4:
        new_file_num = "0" + new_file_num
    with open(f"/Users/davidzoro/RDE1D/_output/fort.q{new_file_num}", "rb") as f:
        for _ in range(5):
            next(f) 
        temp_data = np.mean(np.loadtxt(f), axis=0)
    data.append(temp_data)
data = np.vstack(data)
for i in range(data.shape[1]):
    if i==0:
        labels = data[:, 0]
        inputs = data[:,1:]
        name = "Density"
    elif i==3:
        labels = data[:, 3]
        inputs = data[:,:3]
        name = "Density of Combustion Progress Variable"
    elif i==1:
        labels = data[:,1]
        inputs = data[:, [0, 2, 3]]
        name = "Momentum Density"
    elif i == 2:
        labels = data[:, 2]
        inputs = data[:, [0, 1, 3]]
        name = "Total Energy Density"


    rows = data.shape[0]

    train_inputs = inputs[:int(.9*(rows)), :]
    train_labels = labels[:int(.9*(rows))]

    validation_inputs = inputs[int(.9*(rows)):, :]
    validation_labels = labels[int(.9*(rows)):]

    # Model
    class UNET (nn.Module):
        def __init__(self, in_channels):
            super(UNET, self).__init__()

            self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels= in_channels * 2, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(in_channels=in_channels * 2,out_channels=  in_channels * 4, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv1d(in_channels=in_channels * 4, out_channels= in_channels * 8, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.ConvTranspose1d(in_channels=in_channels * 8,out_channels=  in_channels * 8, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv1d(in_channels=in_channels * 8, out_channels= in_channels * 16, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv1d(in_channels=in_channels * 16, out_channels= in_channels * 32, kernel_size=3, stride=1, padding=1)
            self.conv7 = nn.Conv1d(in_channels=in_channels * 32, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
            self.conv8= nn.ConvTranspose1d(in_channels=in_channels * 64, out_channels= in_channels * 64, kernel_size=3, stride=1, padding=1)
            self.conv9 = nn.Conv1d(in_channels=in_channels * 64, out_channels= in_channels * 128, kernel_size=3, stride=1, padding=1)
            self.conv10 = nn.Conv1d(in_channels=in_channels * 128, out_channels= in_channels * 256, kernel_size=3, stride=1, padding=1)
            self.conv11 = nn.Conv1d(in_channels=in_channels * 256, out_channels= in_channels * 512, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_len, features) to (batch_size, features, seq_len)
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
            x = self.conv11(x)
            x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, features)
            return x

    class Transformer(nn.Module):
        def __init__(self, in_channels, d_model, num_heads, num_encoder_layers, num_classes):
            super(Transformer, self).__init__()
            self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads), num_encoder_layers)
            self.regressor = nn.Linear(1536, num_classes)
            self.the_UNET = UNET(in_channels)
        def forward(self, x):
            x = self.TransformerEncoder(x)
            x = x.reshape(1, x.shape[0], x.shape[1])
            x = self.the_UNET(x)
            x = self.regressor(x)
            return x

    #Define hyperparameters
    epochs = 500
    loss_func = nn.MSELoss()
    Model = Transformer(in_channels=3, d_model=3, num_heads=1, num_encoder_layers=2, num_classes=1)
    learning_rate = 0.0001
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
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_predictions = Model(validation_inputs)
        loss = loss_func(val_predictions, validation_labels)

    print("Validation Loss:", loss.item())
    print("Training complete")
    plt.plot(np.linspace(0, len(train_loss)-1,  len(train_loss)), train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Mean-Squared Error")
    plt.show()
    print("Validation Predictions Shape:", val_predictions.shape, val_predictions)
    print("Validation Labels Shape:", validation_labels.shape)
    new_predictions = val_predictions.squeeze().numpy()

    # Predicted by ground truth
    plt.plot(np.linspace(0, 90 + (len(new_predictions)-1)/10, len(new_predictions)+len(train_labels)), np.concatenate([train_labels, new_predictions]), label="Predicted", color="red")
    plt.plot(np.linspace(0, 90 + (len(new_predictions)-1)/10, len(new_predictions)+len(train_labels)), np.concatenate([train_labels, validation_labels]), label="Ground Truth", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{name}")
    plt.show()
    plt.figure()

    plt.plot(np.linspace(90, 90 + (len(new_predictions)-1)/10, len(new_predictions)), new_predictions, label="Predicted", color="red")
    plt.plot(np.linspace(90, 90 + (len(new_predictions)-1)/10, len(new_predictions)), validation_labels, label="Ground Truth", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{name}")
    plt.show()


    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    # Assuming `new_predictions` and `validation_labels` are already defined
    time = np.linspace(90, 90 + (len(new_predictions) - 1) / 10, len(new_predictions))

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(90, 90 + (len(new_predictions) - 1) / 10)
    ax.set_ylim(min(new_predictions.min(), validation_labels.min()), max(new_predictions.max(), validation_labels.max()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{name}")

    # Initialize lines for predicted and ground truth
    predicted_line, = ax.plot([], [], label="Predicted", color="red")
    ground_truth_line, = ax.plot([], [], label="Ground Truth", color="blue")
    ax.legend()

    # Update function for animation
    def update(frame):
        predicted_line.set_data(time[:frame], new_predictions[:frame])
        ground_truth_line.set_data(time[:frame], validation_labels[:frame])
        return predicted_line, ground_truth_line

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(time), interval=100, blit=True)

    # Save the animation as a video
    ani.save(f"{name}_1fps.mp4", writer="ffmpeg", fps=1)

    plt.show()
    
    
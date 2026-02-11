import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, criterion, lr, l2, epochs):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(lr=lr, params=self.parameters(), weight_decay=l2)
        self.epochs = epochs

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

    def fit(self, x_train, y_train, x_val, y_val):
        for epoch in range(self.epochs):
            self.train()
            outputs = self(x_train)
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.inference_mode():
                # 1. Forward pass
                self.eval()
                val_outputs = self(x_val)
                val_outputs = val_outputs.squeeze()
                val_loss = self.criterion(val_outputs, y_val)
        return val_loss

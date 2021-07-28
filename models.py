import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 16, 3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv6 = nn.Conv1d(32, 32, 3, padding=1, stride=2)
        self.conv7 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv8 = nn.Conv1d(64, 64, 3, padding=1, stride=2)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)
        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)
        self.p3 = nn.MaxPool1d(5)
        self.p4 = nn.MaxPool1d(5)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = wave.view(-1, 1, 6000)
        wave = self.bn1(F.relu(self.conv1(wave)))
        wave = self.bn2(F.relu(self.conv2(wave)))
        wave = self.p1(wave)
        wave = self.bn3(F.relu(self.conv3(wave)))
        wave = self.bn4(F.relu(self.conv4(wave)))
        wave = self.p2(wave)
        wave = self.bn5(F.relu(self.conv5(wave)))
        wave = self.bn6(F.relu(self.conv6(wave)))
        wave = self.p3(wave)
        wave = self.bn7(F.relu(self.conv7(wave)))
        wave = self.bn8(F.relu(self.conv8(wave)))
        wave = self.p4(wave)
        wave = wave.squeeze()
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)

class CRED(nn.Module):
    def __init__(self):
        super(CRED, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=4)
        self.res1conv1 = nn.Conv1d(8, 8, 3, padding=1, stride=1)
        self.res1conv2 = nn.Conv1d(8, 8, 3, padding=1, stride=1)

        self.conv2 = nn.Conv1d(8, 16, 3, padding=1, stride=4)
        self.res2conv1 = nn.Conv1d(16, 16, 3, padding=1, stride=1)
        self.res2conv2 = nn.Conv1d(16, 16, 3, padding=1, stride=1)

        self.bilstm1 = nn.LSTM(375, 64, 1, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)

        self.dropbi = nn.Dropout(p=0.7)
        self.droplstm = nn.Dropout(p=0.8)

        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 1)

        self.bn1 = nn.BatchNorm1d(8)
        self.res1bn1 = nn.BatchNorm1d(8)
        self.res1bn2 = nn.BatchNorm1d(8)

        self.bn2 = nn.BatchNorm1d(16)
        self.res2bn1 = nn.BatchNorm1d(16)
        self.res2bn2 = nn.BatchNorm1d(16)

        self.bnlstm = nn.BatchNorm1d(16)
        self.bnlinear = nn.BatchNorm1d(128)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, 6000)

        # Primera capa CNN
        x = self.conv1(x)
        identity = x

        # Residual block 1
        x = F.relu(self.bn1(x))
        x = F.relu(self.res1bn1(self.res1conv1(x)))
        x = F.relu(self.res1bn2(self.res1conv2(x)))
        x = x + identity

        # Segunda capa CNN
        x = self.conv2(x)
        identity = x

        # Residual block 2
        x = F.relu(self.bn2(x))
        x = F.relu(self.res2bn1(self.res2conv1(x)))
        x = F.relu(self.res2bn2(self.res2conv2(x)))
        x = x + identity

        # Time redistribution
        # x = x.permute(0, 2, 1)

        # Bi LSTM
        x, _ = self.bilstm1(x)
        x = self.bnlstm(x)
        x = self.dropbi(x)

        # Bi LSTM residual block
        identity = x
        x, _ = self.bilstm2(x)
        x = x + identity
        x = self.bnlstm(x)
        x = self.dropbi(x)

        # LSTM
        x, _ = self.lstm(x)
        x = self.bnlstm(x)
        x = self.droplstm(x)

        # Linear
        x = self.l1(x[:, -1, :])
        x = self.bnlinear(x)
        x = self.droplstm(x)

        x = self.l2(x)

        return self.sigmoid(x)
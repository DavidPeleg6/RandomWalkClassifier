from gcommand_loader import GCommandLoader
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
import torch
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 20
learning_rate = 0.001
conv_out = 20
num_blocks = [2]*3
num_classes = 30
p = 0.5


def convfilter3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convfilter3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convfilter3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, classes=30):
        super(ResNet, self).__init__()
        self.in_channels = 20
        self.conv = convfilter3(1, 20)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.create_res_layer(block, out_channels=20, blocks=layers[0])
        self.layer2 = self.create_res_layer(block, out_channels=40, blocks=layers[1], stride=2)
        self.layer3 = self.create_res_layer(block, out_channels=60, blocks=layers[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(900, 200)
        self.drop = F.dropout(p=p)
        self.fc2 = nn.Linear(200, classes)

    def create_res_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                convfilter3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


def train(train_loader, model, optimizer):
    model.train()
    running_loss = 0
    correct_train = 0
    for batch_idx, (sound, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        sound = sound.to(device)
        labels = labels.to(device)
        output = model.forward(sound)
        loss = nn.functional.nll_loss(output, labels, reduction='sum')
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = output.max(1, keepdim=True)[1]
        correct_train += predicted.eq(labels.view_as(predicted)).cpu().sum()
    return running_loss/(len(train_loader)*int(batch_size)), 100. * correct_train / (len(train_loader)*int(batch_size))

def val(validation_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        v_loss = 0
        for batch_idx, (sound, labels) in enumerate (validation_loader):
            sound = sound.to(device)
            labels = labels.to(device)
            output = model(sound)
            v_loss += nn.functional.nll_loss(output, labels, reduction='sum')
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum()
    return v_loss/(len(validation_loader)*int(batch_size)), 100. * correct / (len(validation_loader)*int(batch_size))


def test_file_output(testset, test_loader, model):
    with open('test_y', 'w') as test_y_file:
        model.eval()
        with torch.no_grad():
            test_files_names = []
            test_pred = []
            for file_name in range(len(testset)):
                test_files_names.append((testset.spects[file_name])[0].split("\\")[2])

            for batch_idx, (sound, _) in enumerate(test_loader):
                sound = sound.to(device)
                output = model(sound)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log prob
                for p in pred:
                    test_pred.append(p.item())
            for prediction, file_name in zip(test_pred, test_files_names):
                test_y_file.write(str(file_name) + ', ' + str(prediction) + '\n')


def main():
    total_t = time.time()

    #load data
    trainset = GCommandLoader('./train')
    testset = GCommandLoader('./test')
    validationset = GCommandLoader('./valid')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, sampler=None)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True, sampler=None)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    model = ResNet(ResidualBlock, num_blocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        t = time.time()

        train_avg_loss, train_avg_acc = train(train_loader, model, optimizer)
        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)

        val_avg_loss, val_avg_acc = val(validation_loader, model)
        val_loss.append(val_avg_loss)
        val_acc.append(val_avg_acc)

        print("Epoch: {}/{}".format(e+1, num_epochs),
              "Train loss: {:.3f}".format(train_avg_loss),
              "Train acc: {:.3f}".format(train_avg_acc),
              "Val loss: {:.3f}".format(val_avg_loss),
              "Val acc: {:.3f}".format(val_avg_acc),
              "time in min: {:.3f}".format((time.time() - t))/60)

    test_file_output(testset, test_loader, model)

    print("total time", (time.time() - total_t)/60)
    e=0




if __name__ == "__main__":
    main()

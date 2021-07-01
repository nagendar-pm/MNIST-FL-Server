import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


class Model:
    def __init__(self):
        self.model = CNN_2().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.setModel()

    def setModel(self):
        # loading the model to be trained
        state = torch.load('Models\\cnn_2.pt')
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def getModel(self):
        return self.model

    def predict_img(self, input):
        input = torch.tensor([input])
        input = torch.reshape(input, (1, 16, 14, 14))
        outputs = self.model(input)[0]
        print(outputs)
        prediction = torch.max(outputs, 1)[1].data.squeeze()
        print("Predicted Digit =", prediction)
        return prediction, outputs

    def backward(self, outputs, label):
        # set the model for training
        self.model.train()
        activations = outputs.detach().requires_grad_(True)
        label = torch.tensor([label])
        loss = self.criterion(activations, label)
        # clear gradients for this training step
        self.optimizer.zero_grad()
        a = self.model.conv2[0].weight.clone()
        # back propagation, compute gradients
        loss.backward()
        outputs.backward(activations.grad)
        # apply gradients
        self.optimizer.step()
        b = self.model.conv2[0].weight.clone()

        print("\na\n")
        print(str(a))
        print("\nb\n")
        print(str(b))
        print(a == b)
        print(a is b)
        self.save()

    def save(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, 'Models\\cnn_2.pt')

        # example = torch.rand(1, 500)
        traced_script_module = torch.jit.script(self.model)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized.save("Models\\updated_model.pt")

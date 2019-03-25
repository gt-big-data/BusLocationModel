import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.linear_layer_1 = nn.Linear(2, 2)
        self.non_linear_func_1 = F.relu
        self.linear_layer_2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.non_linear_func_1(x)
        x = self.linear_layer_2(x)
        return x

class MyLossFunction(nn.Module):
    def __init__(self):
        super(MyLossFunction, self).__init__()

    def forward(self, model_output, target_output):
        loss = (model_output - target_output) ** 2
        return loss

# let the model approximate the averaging function
examples = [
    [0, 0,  0],
    [0, 1,  0.5],
    [1, 0,  0.5],
    [1, 1,  1],
]

model = MyClassifier()
criterion = MyLossFunction()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train() # train mode
running_loss = 0.0
for epoch in range(20000):
    random.shuffle(examples)
    for example in examples:
        # prepare the data
        input_features = torch.FloatTensor(example[:2])
        target_output = torch.FloatTensor(example[2:])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input_features) # forward (compute model output)
        loss = criterion(output, target_output)
        loss.backward() # backward (compute parameter gradients)
        optimizer.step() # update parameters

        running_loss += loss.item()
    if epoch % 2000 == 1999:
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
        running_loss = 0.0

model.eval() # test mode
test_input = [0, 0]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [0, 1]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [1, 0]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [1, 1]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [0.1, 1.1]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [-0.1, 0.1]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [0.9, 1.1]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

test_input = [2019, 2020]
test_input_features = torch.FloatTensor(test_input)
print(test_input, model(test_input_features).item())

import torch
import torch.nn as nn


# Define a simple linear layer
class QModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(QModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# Define input and output sizes
input_size = 10
output_size = 2

# Create an instance of the model
model = QModel(input_size, output_size)

# Generate some random input data
state_1 = torch.randn(1, 5, input_size)
state_2 = torch.randn(1, 10, input_size)

replay_buffer_state = [state_1, state_2]
replay_buffer_reward = [1.1, 2.2]
replay_buffer_action = [0, 1]

# print("Input data:")
# print(input_data.shape)
print("Output data:")
#print(output_data.shape)

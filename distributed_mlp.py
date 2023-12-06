import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__ (self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
        super().__init__()

        # first layer
        layers = [
                nn.Linear(in_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                # nn.Dropout(p=0.5)
                
        ]

        # last layer
        out_layer_list = [
            nn.Linear(hidden_feature, out_feature),
            nn.Sigmoid()
        ]

        # hidden layers
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(hidden_feature, hidden_feature),
                nn.BatchNorm1d(hidden_feature),
                nn.ReLU()
            ])
        
        layers.extend(out_layer_list)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# class MLP(nn.Module):
#     def __init__(self, in_feature: int, hidden_feature: int, hidden_layers: int, out_feature: int):
#         super().__init__()

#         # Split the hidden layers into two parts
#         half_hidden_layers = hidden_layers // 2

#         # First half of the model
#         layers_first_half = [
#             nn.Linear(in_feature, hidden_feature),
#             nn.BatchNorm1d(hidden_feature),
#             nn.ReLU()
#         ]
#         for _ in range(half_hidden_layers):
#             layers_first_half.extend([
#                 nn.Linear(hidden_feature, hidden_feature),
#                 nn.BatchNorm1d(hidden_feature),
#                 nn.ReLU()
#             ])

#         # Second half of the model
#         layers_second_half = []
#         for _ in range(half_hidden_layers, hidden_layers):
#             layers_second_half.extend([
#                 nn.Linear(hidden_feature, hidden_feature),
#                 nn.BatchNorm1d(hidden_feature),
#                 nn.ReLU()
#             ])

#         # Last layer
#         layers_second_half.extend([
#             nn.Linear(hidden_feature, out_feature),
#             nn.Sigmoid()
#         ])

#         self.first_half = nn.Sequential(*layers_first_half).to('cuda:0')
#         self.second_half = nn.Sequential(*layers_second_half).to('cuda:1')

#     def forward(self, x):
#         # Transfer x to cuda:0
#         x = x.to('cuda:0')
#         x = self.first_half(x)
#         # Transfer x to cuda:1 for processing by the second half
#         x = x.to('cuda:1')
#         x = self.second_half(x)
#         return x








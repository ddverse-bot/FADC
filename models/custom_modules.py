 self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction_ratio, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class FADC_With_Adakern_Internal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rates=[1, 2, 4], reduction_ratio=4):
        super(FADC_With_Adakern_Internal, self).__init__()
        self.dilation_rates = dilation_rates
        self.num_branches = len(dilation_rates)
        self.out_channels = out_channels

        self.dilated_convs = nn.ModuleList()
        for dilation in dilation_rates:
            padding = dilation * (kernel_size - 1) // 2
            self.dilated_convs.append(
                Adakern_DilatedLayer(in_channels, out_channels, kernel_size, padding, dilation)
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_dim, self.num_branches * out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight_sharpening_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        branch_outputs = [conv_branch(x) for conv_branch in self.dilated_convs]

        weights = self.avg_pool(x)
        weights = self.fc1(weights)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = weights * self.weight_sharpening_scale
        weights = self.sigmoid(weights)

        weights = weights.view(-1, self.num_branches, self.out_channels, 1, 1)

        combined_output = torch.zeros_like(branch_outputs[0])
        for i, output in enumerate(branch_outputs):
            combined_output += output * weights[:, i, :, :, :]
        return combined_output

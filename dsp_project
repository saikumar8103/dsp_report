import torch
import torch.nn as nn

class TimeVaryingIIR(nn.Module):
    def __init__(self, order=2):
        super(TimeVaryingIIR, self).__init__()
        self.order = order

        self.b_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, order + 1)
        )

        self.a_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, order)
        )

    def forward(self, x):
        batch, T = x.shape
        y = torch.zeros_like(x)

        x_past = torch.zeros(batch, self.order, device=x.device)
        y_past = torch.zeros(batch, self.order, device=x.device)

        for t in range(T):
            xt = x[:, t].unsqueeze(-1)

            b = self.b_net(xt)
            a = torch.tanh(self.a_net(xt))

            yt = b[:, 0] * xt.squeeze(-1)

            for i in range(self.order):
                yt += b[:, i + 1] * x_past[:, i]
                yt -= a[:, i] * y_past[:, i]

            y[:, t] = yt

            x_past = torch.roll(x_past, shifts=1, dims=1)
            x_past[:, 0] = xt.squeeze(-1)

            y_past = torch.roll(y_past, shifts=1, dims=1)
            y_past[:, 0] = yt

        return y


class SpeechDenoiser(nn.Module):
    def __init__(self):
        super(SpeechDenoiser, self).__init__()
        self.iir = TimeVaryingIIR(order=2)

    def forward(self, noisy_signal):
        return self.iir(noisy_signal)


# Example usage
if __name__ == "__main__":
    model = SpeechDenoiser()
    noisy = torch.randn(1, 16000)  # 1 sec audio @16kHz
    output = model(noisy)
    print(output.shape)

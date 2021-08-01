from torch import nn, optim
import torch

device='cuda'
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()

        self.lstm1=nn.LSTM(input_dim, hidden_size=hidden_dim//2,batch_first=True, bidirectional=True)
        self.lstm2=nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.lstm3=nn.LSTM(hidden_dim*2, hidden_size=hidden_dim//2, batch_first=True, bidirectional=True)

        self.lstm_d1=nn.LSTM(hidden_dim+vocab_size, hidden_dim, batch_first=True)
        self.lstm_d2=nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear=nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size

        self.att_score=nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def step_one_layer(self, input, lstm):
        # temp = None
        out, _ = lstm(input)
        add_mask = torch.stack([torch.zeros((out.shape[1] // 2,)), torch.ones((out.shape[1] // 2,))], dim=-1).view(-1)
        if out.shape[1] % 2:
            # out, temp = torch.split(out, (-1, 1), dim=1)
            out = out[:, :-1]
        flag = add_mask.bool()

        input2 = out[:, flag] + out[:, torch.bitwise_not(flag)]
        # if temp is not None:
        #     input2 = torch.cat([input2, temp], dim=1)

        return input2

    def forward(self, x, y, pad_flag):
        x=self.step_one_layer(x, self.lstm1)
        x=self.step_one_layer(x, self.lstm2)
        x=self.step_one_layer(x, self.lstm3) # (B, L, hidden_dim)

        h = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        c = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        a1s=[]
        # print(self.vocab_size, y.max())

        one_hot = torch.zeros(y.shape+(self.vocab_size,), device=device).scatter(-1, y.unsqueeze(-1), 1)

        for i in range(y.shape[1]):
            att_i = torch.cat(torch.broadcast_tensors(x, h.view(x.shape[0],1,-1)), dim=-1)
            att_score = torch.softmax(self.att_score(att_i), dim=1) # (B, L, 1)

            input_dec=(att_score*x).sum(dim=1, keepdim=True) # (B, 1, hidden_dim)
            out, [h, c]=self.lstm_d1(torch.cat([input_dec, one_hot[:, i:i+1]], dim=-1), [h, c])

            a1s.append(out)

        a2s, _ = self.lstm_d2(torch.cat(a1s, dim=1))

        out = self.linear(a2s)

        loss = self.nll_loss(out, y, pad_flag)

        return out, loss

    def nll_loss(self, y_, y, flag):
        return -(torch.log(torch.gather(torch.softmax(y_, dim=-1), -1, y.unsqueeze(-1))+1e-8)*flag.unsqueeze(-1)).sum()/(flag.sum()+1e-8)

    def mask_acc(self, y_, y, flag):
        return ((y_.argmax(dim=-1)==y).int()*flag).sum()/(flag.sum()+1e-8)


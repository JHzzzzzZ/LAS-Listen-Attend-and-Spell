from torch import nn, optim
import torch

device='cuda'
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        '''
        模型初始化，各个层的定义
        :param input_dim:[int] 输入的feature维度
        :param hidden_dim:[int] LSTM层的hidden_dim
        :param vocab_size:[int] 标签词表大小
        '''
        super().__init__()

        # encoder
        self.lstm1=nn.LSTM(input_dim, hidden_size=hidden_dim//2,batch_first=True, bidirectional=True)
        self.lstm2=nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.lstm3=nn.LSTM(hidden_dim*2, hidden_size=hidden_dim//2, batch_first=True, bidirectional=True)

        # decoder
        self.lstm_d1=nn.LSTM(hidden_dim+vocab_size, hidden_dim, batch_first=True)
        self.lstm_d2=nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear=nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size

        # attention
        self.att_score=nn.Sequential(
            nn.Linear(hidden_dim+hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def step_one_layer(self, input, lstm):
        '''
        相邻两个位置的out相加
        :param input: [tensor] lstm层的输入 (B, L, feature_dim)
        :param lstm: [module] 具体是哪个lstm层
        :return: pyramid之后的, 下一层的input, (B, L//2, out_dim)
        '''
        # temp = None
        out, _ = lstm(input)
        # [0,1,0,..., 1] 标志
        add_mask = torch.stack([torch.zeros((out.shape[1] // 2,)), torch.ones((out.shape[1] // 2,))], dim=-1).view(-1)
        # 如果长度为奇数，去掉最后一个out
        if out.shape[1] % 2:
            # out, temp = torch.split(out, (-1, 1), dim=1)
            out = out[:, :-1]
        # 转化为[False, True, False, ..., True]
        flag = add_mask.bool()

        # 索引相加
        input2 = out[:, flag] + out[:, torch.bitwise_not(flag)]
        # if temp is not None:
        #     input2 = torch.cat([input2, temp], dim=1)

        return input2

    def forward(self, x, y, pad_flag):
        '''
        前向传播
        :param x:[tensor] (B, L, feature_dim) 输入的acoustic feature
        :param y:[tensor] (B, L') 输入对应的输出
        :param pad_flag:[tensor] (B, L') 输出对应的pad标志位
        :return: 前向传播的结果与loss
        '''
        # encode
        x=self.step_one_layer(x, self.lstm1)
        x=self.step_one_layer(x, self.lstm2)
        x=self.step_one_layer(x, self.lstm3) # (B, L, hidden_dim)

        # attention计算的准备
        h = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        c = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        a1s=[]

        # 加入<bos>去掉<eos> bos_id=0
        temp_y = torch.cat([torch.zeros((y.shape[0],1), dtype=torch.long, device=device), y[:, :-1]], dim=-1)
        one_hot = torch.zeros(y.shape+(self.vocab_size,), device=device).scatter(-1, temp_y.unsqueeze(-1), 1)

        # decoder-1与attention计算
        for i in range(y.shape[1]):
            # 拼接h_i与x (B, L, hidden_dim*2)
            att_i = torch.cat(torch.broadcast_tensors(x, h.view(x.shape[0],1,-1)), dim=-1)
            # 计算attention_score
            att_score = torch.softmax(self.att_score(att_i), dim=1) # (B, L, 1)

            # 加权相加
            input_dec=(att_score*x).sum(dim=1, keepdim=True) # (B, 1, hidden_dim)
            # 解码得到输出
            out, [h, c]=self.lstm_d1(torch.cat([input_dec, one_hot[:, i:i+1]], dim=-1), [h, c])
            # 记录结果
            a1s.append(out)
        # decoder-2
        a2s, _ = self.lstm_d2(torch.cat(a1s, dim=1))
        # decoder-linear
        out = self.linear(a2s)

        loss = self.nll_loss(out, y, pad_flag)

        return out, loss

    def nll_loss(self, y_, y, flag):
        '''
        计算cross entropy。
        :param y_: [tensor] (B, L', vocab_size) 模型的输出
        :param y: [tensor] (B, L') 标签
        :param flag: [tensor] (B, L') pad标志位
        :return: masked cross entropy
        '''
        return -(torch.log(torch.gather(torch.softmax(y_, dim=-1), -1, y.unsqueeze(-1))+1e-8)*flag.unsqueeze(-1)).sum()/(flag.sum()+1e-8)

    def mask_acc(self, y_, y, flag):
        '''
        计算acc。
        :param y_: [tensor] (B, L', vocab_size) 模型的输出
        :param y: [tensor] (B, L') 标签
        :param flag: [tensor] (B, L') pad标志位
        :return: masked acc
        '''
        return ((y_.argmax(dim=-1)==y).int()*flag).sum()/(flag.sum()+1e-8)


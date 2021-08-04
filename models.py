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
        self.cnn1=nn.Conv1d(input_dim, hidden_dim//2, kernel_size=7, padding=3, )
        self.ln1=nn.LayerNorm(hidden_dim//2)
        self.lstm1=nn.LSTM(hidden_dim//2, hidden_size=hidden_dim//2,batch_first=True, bidirectional=True)
        self.cnn2=nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.lstm2=nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.cnn3 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.lstm3=nn.LSTM(hidden_dim, hidden_size=hidden_dim//2, batch_first=True, bidirectional=True)

        self.act=nn.LeakyReLU(0.1)

        # decoder
        self.lstm_d1=nn.LSTM(hidden_dim+vocab_size, hidden_dim, batch_first=True)
        self.lstm_d2=nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear=nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size

        # attention
        # self.att_score=nn.Sequential(
        #     nn.Linear(hidden_dim+hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, 1),
        # )

    def step_one_layer(self, input, lstm, cnn, ln):
        '''
        相邻两个位置的out相加
        :param input: [tensor] lstm层的输入 (B, L, feature_dim)
        :param lstm: [module] 具体是哪个lstm层
        :return: pyramid之后的, 下一层的input, (B, L//2, out_dim)
        '''
        # temp = None
        out = cnn(input.permute(0, 2, 1))
        out = ln(out.permute(0, 2, 1))
        out = self.act(out)

        out, _ = lstm(out)
        # [0,1,0,..., 1] 标志
        add_mask = torch.stack([torch.zeros((out.shape[1] // 2,)), torch.ones((out.shape[1] // 2,))], dim=-1).view(-1)
        # 如果长度为奇数，去掉最后一个out
        if out.shape[1] % 2:
            # out, temp = torch.split(out, (-1, 1), dim=1)
            out = out[:, :-1]
        # 转化为[False, True, False, ..., True]
        flag = add_mask.bool()

        # 索引相加
        # input2 = torch.cat([out[:, flag], out[:, torch.bitwise_not(flag)]], dim=-1)
        input2 = out[:, flag]+out[:, torch.bitwise_not(flag)]
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
        x=self.step_one_layer(x, self.lstm1, self.cnn1, self.ln1)
        x=self.step_one_layer(x, self.lstm2, self.cnn2, self.ln2)
        x=self.step_one_layer(x, self.lstm3, self.cnn3, self.ln3) # (B, L, hidden_dim)

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
            # att_i = torch.cat(torch.broadcast_tensors(x, h.view(x.shape[0],1,-1)), dim=-1)
            # 计算attention_score
            # att_score = torch.softmax(self.att_score(att_i), dim=1) # (B, L, 1)

            att_score=torch.softmax((h.permute(1,0,2)@x.permute(0,2,1)).squeeze(1)/torch.sqrt(
                torch.tensor(x.shape[2],dtype=torch.float,device=device)), dim=-1)

            # 加权相加
            # print(x.shape, att_score.shape,h.permute(1,0,2).shape,x.permute(0,2,1).shape)
            input_dec=(att_score.unsqueeze(-1)*x).sum(dim=1, keepdim=True) # (B, 1, hidden_dim)
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

    def predict(self, x, beam_size=2):
        '''
        根据acoustic feature预测说了啥
        :param x: (1, L, feature_dim)
        :param beam_size: [int] beam search的束大小
        :return: words
        '''
        # encoder
        x = self.step_one_layer(x, self.lstm1)
        x = self.step_one_layer(x, self.lstm2)
        x = self.step_one_layer(x, self.lstm3)  # (B, L, hidden_dim)

        h = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        c = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)

        # prev数组，记录前一个的idx，初始为从<bos>(0)开始
        prev = [[0]]*beam_size

        att_i = torch.cat(torch.broadcast_tensors(x, h.view(x.shape[0], 1, -1)), dim=-1)
        # 计算attention_score
        att_score = torch.softmax(self.att_score(att_i), dim=1)  # (1, L, 1)
        # 加权相加
        input_dec = (att_score * x).sum(dim=1, keepdim=True)  # (1, 1, hidden_dim)

        # 输入<bos>
        p=torch.zeros((x.shape[0], 1))
        one_hot = self.get_one_hot(p)

        # 解码得到输出 (1, 1, hidden_dim)
        out, [h, c] = self.lstm_d1(torch.cat([input_dec, one_hot], dim=-1), [h, c])
        # 记录结果 (1,1,hidden_dim)
        a2, [h_, c_] = self.lstm_d2(out)
        prob_dist = self.linear(a2).squeeze() # (vocab_size,)

        # 找到topk
        score, index=torch.topk(prob_dist, k=beam_size)
        # 暂存index，它将生成下一个的输入
        temp=index

        # 隐藏层状态暂存，第一次预测过后全部是一样的
        hs=[[h,c]]*beam_size
        h_s=[[h_,c_]]*beam_size

        # 最多预测50个字
        # TODO:<eos>结束
        for i in range(50):
            # 解码得到输出 (1, 1, hidden_dim)
            temp_prob=[]
            temp_h=[]
            temp_h_=[]

            bit_map=[]
            # run beam size次
            for idx in temp:
                if idx in bit_map:
                    # 占位，使得bit_map里面找出来的i与idx一致
                    bit_map.append('__')
                    i=bit_map.index(idx)
                    temp_prob.append(temp_prob[i])
                    temp_h.append(temp_h[i])
                    temp_h_.append(temp_h_[i])
                    continue

                att_i = torch.cat(torch.broadcast_tensors(x, hs[idx][0].view(x.shape[0], 1, -1)), dim=-1)
                # 计算attention_score
                att_score = torch.softmax(self.att_score(att_i), dim=1)  # (1, L, 1)
                # 加权相加
                input_dec = (att_score * x).sum(dim=1, keepdim=True)  # (1, 1, hidden_dim)
                one_hot=self.get_one_hot(torch.full((x.shape[0], 1), idx))
                out, [h, c] = self.lstm_d1(torch.cat([input_dec, one_hot], dim=-1), hs[idx])
                # 记录结果 (1,1,hidden_dim)
                a2, [h_, c_] = self.lstm_d2(out, h_s[idx])
                prob_dist=self.linear(a2)

                # 用来cat+score找topk
                temp_prob.append(prob_dist)
                # 用来记录下一波hs和h_s
                temp_h.append([h, c])
                temp_h_.append([h_,c_])

                bit_map.append(idx)

            # 把预测的beam_size个都拿过来拼接起来加上score找topk
            s=torch.stack(temp_prob, dim=0)+score.unsqueeze(-1)
            score, indices=torch.topk(s.view(-1), k=beam_size)
            idxs=[]
            t=[]
            hs=[]
            h_s=[]
            for each in indices:
                # indices的内容符合->[...(beam1), ...(beam2), ......],所以整除过后就是prev
                index=each//self.vocab_size
                idxs.append(temp[index])
                # t将作为temp控制下一波prob_dist的计算
                t.append(each%self.vocab_size)
                # 更新hs与h_s
                hs.append(temp_h[index])
                h_s.append(temp_h_[index])

            prev.append(idxs)
            temp=t

        return prev, score

    def get_one_hot(self, y):
        return torch.zeros((y.shape[0], 1, self.vocab_size), device=device).scatter(-1, y.unsqueeze(-1), 1)

    def predict_(self, x, eos_id=1):
        '''
        根据acoustic feature预测说了啥
        :param x: (1, L, feature_dim)
        :param beam_size: [int] beam search的束大小
        :return: words
        '''
        # encoder
        x = self.step_one_layer(x, self.lstm1, self.cnn1, self.ln1)
        x = self.step_one_layer(x, self.lstm2, self.cnn2, self.ln2)
        x = self.step_one_layer(x, self.lstm3, self.cnn3, self.ln3)  # (B, L, hidden_dim)

        h = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        c = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        h_ = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
        c_ = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)

        # 输入<bos>
        p = torch.zeros((x.shape[0], 1), dtype=torch.long, device=device)
        one_hot = self.get_one_hot(p)

        ans=[]

        # 最多预测50个字
        # TODO:<eos>结束
        for i in range(50):
        # 解码得到输出 (1, 1, hidden_dim)
        #     att_i = torch.cat(torch.broadcast_tensors(x, h.view(x.shape[0], 1, -1)), dim=-1)
            # 计算attention_score
            # att_score = torch.softmax(self.att_score(att_i), dim=1)  # (1, L, 1)
            # # 加权相加

            att_score = torch.softmax((h.permute(1, 0, 2) @ x.permute(0, 2, 1)).squeeze(1) / torch.sqrt(
                torch.tensor(x.shape[2], dtype=torch.float, device=device)), dim=-1)
            input_dec = (att_score.unsqueeze(-1) * x).sum(dim=1, keepdim=True)  # (1, 1, hidden_dim)

            # 解码得到输出 (1, 1, hidden_dim)
            out, [h, c] = self.lstm_d1(torch.cat([input_dec, one_hot], dim=-1), [h, c])
            # 记录结果 (1,1,hidden_dim)
            a2, [h_, c_] = self.lstm_d2(out, [h_, c_])
            prob_dist = self.linear(a2).squeeze() # (vocab_size,)
            ans.append(prob_dist.argmax())
            if ans[-1]==eos_id:
                break
            one_hot = self.get_one_hot(torch.full((x.shape[0], 1), ans[-1], dtype=torch.long, device=device))

        return ans




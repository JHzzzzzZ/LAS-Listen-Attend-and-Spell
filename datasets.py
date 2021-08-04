import torchaudio as ta
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
import pickle
import audiomentations
import h5py

def count_txt(txt_paths):
    '''
    统计标签的特征
    :param txt_paths:[list] 标签的路径
    :return: None
    '''
    mn=1000000
    mx=-1
    avg=0
    c=0
    d={}
    for file in tqdm(txt_paths):
        with open(file, encoding='utf8') as f:
            for line in f:
                line=line.rstrip('\n')
                if line:
                    length=len(line)
                    d[length]=d.get(length, 0)+1
                    mx=max(mx, length)
                    mn=min(mn, length)
                    avg+=length
                    c+=1

    print(f'最长句子长度：{mx}, 最短句子长度：{mn}, 平均句子长度：{avg/c}')
    plt.bar(d.keys(), d.values())
    plt.show()


def count_feature(wav_paths):
    '''
    统计wav的特征
    :param wav_paths:[list] wav文件列表
    :return: None
    '''
    mn=1000000
    mx=-1
    avg=0
    c=0
    d={}
    for file in tqdm(wav_paths):
        wave_form, sample_freq=ta.load(file)
        # length=ta.compliance.kaldi.mfcc(wave_form, sample_frequency=sample_freq).shape[0]

        # mfcc的长度大致是wave_form的长度除以160
        length=wave_form.shape[1]//160
        mx=max(mx, length)
        d[length]=d.get(length, 0)+1
        mn = min(mn, length)
        avg+=length
        c+=1

    print(f'最长语音特征长度：{mx}, 最短语音特征长度：{mn}, 平均语音特征长度：{avg/c}')
    plt.bar(d.keys(), d.values())
    plt.show()

class Data(Dataset):
    def __init__(self, wav_paths, txt_paths, train=True, vocab=None, save_path='./vocab_label.pkl',
                 num_data=100, feature_save_path='./data'):
        '''
        数据集初始化与标签读取
        :param wav_paths:[list] wav文件列表
        :param txt_paths:[list] txt文件列表 顺序应该与wav文件一致
        :param train:[bool] 训练集还是验证集
        :param vocab:Union[None, dict] 如果train=True, 则vocab=None, 如果train=False，则vocab=train_dataset.vocab
        :param save_path:[str] 保存训练集的已经读取的文件与标签
        '''
        super().__init__()

        assert train or vocab is not None, 'validation时需要传入vocab'

        self.wav_paths=wav_paths
        self.txt_paths=txt_paths
        self.num_data=num_data
        self.feature_save_path=feature_save_path
        self.train=train

        self.augment = audiomentations.Compose([
                audiomentations.AddGaussianNoise(),
                audiomentations.TimeStretch(), # 对时间维度调整
                audiomentations.PitchShift(), # 对音调调整
                audiomentations.Shift(), # 在时间轴的滚动，主要是用到np.roll()函数
            ])

        if train and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.vocab, self.labels=pickle.load(f)
        else:
            lines=[]
            self.words=[]
            for file in tqdm(self.txt_paths):
                with open(file, encoding='utf8') as f:
                    for line in f:
                        line=line.strip('\n')
                        if line:
                            lines.append(line)
                            self.words.extend(list(line))

            if train:
                self.words=['<bos>', '<eos>', '<pad>', '<unk>'] + list(sorted(set(self.words)))

                self.vocab={v:k for k,v in enumerate(self.words)}
            else:
                self.vocab=vocab

            self.labels=[]
            for line in lines:
                self.labels.append([self.vocab.get(each, self.vocab['<unk>']) for each in line])

            if train:
                with open(save_path, 'wb') as f:
                    pickle.dump([self.vocab, self.labels], f)
        assert len(self.labels)==len(self.wav_paths), '%s %s'%(len(self.labels), len(self.wav_paths))
        self.data=[]
        self.s=0
        self.idx=[]
        # self.load_data_in_one_time()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        返回acoustic feature与对应标签
        :param idx:[int] 索引
        :return:(feature, label)
        '''
        # 分段load
        if idx==0:
            self.s=0
        if idx >= self.s:
            self.load_data(self.wav_paths[self.s:min(self.s+self.num_data, len(self.wav_paths))])
            self.s+=self.num_data

        i=idx-self.s+self.num_data
        # print(i, idx, self.s, len(self.data))
        # 逆序
        # return self.data[self.idx[idx][0]-self.s+self.num_data], self.labels[self.idx[idx][0]]
        # 伪随机
        return self.data[i], self.labels[idx]

        # load_data_in_one_time 用这一行
        # return self.data[idx], self.labels[idx]

    def load_data_in_one_time(self):
        for s in tqdm(range(0, len(self), self.num_data)):
            if self.train:
                file = os.path.join(self.feature_save_path, '%3d-%3d.h5' % (s, s + self.num_data))
            else:
                file = os.path.join(self.feature_save_path, 'val_%3d-%3d.h5' % (s, s + self.num_data))

            with open(file, 'rb') as f:
                data, idx=pickle.load(f)
            self.data.extend(data)
            self.idx.extend(idx)
            self.s+=self.num_data

        assert len(self.labels)==len(self.data), '%d %d'%(len(self.labels), (len(self.data)))

    def h5save(self, filename, **kwargs):
        with h5py.File(filename, 'w') as f:
            for k,v in kwargs.items():
                g = f.create_group(k)
                if isinstance(v, (list, tuple)):
                    for i, each in enumerate(v):
                        g.create_dataset('%03d'%i, data=each, compression='gzip', chunks=True)
                else:
                    g.create_dataset(k, data=v, compression='gzip', chunks=True)

    def h5load(self, filename):
        d={}
        with h5py.File(filename, 'r') as f:
            for k,v in f.items():
                if v.keys():
                    d[k]=[]
                    for kk, v in v.items():
                        d[k].append(torch.tensor(v[:]))
                else:
                    d[k]=torch.tensor(v[:])
                # print(v.keys())
            return d

    def load_data(self, paths):
        self.data=[]
        self.idx=[]
        if self.train:
            file = os.path.join(self.feature_save_path, '%d-%d.dat'%(self.s, self.s+self.num_data))
        else:
            file = os.path.join(self.feature_save_path, 'val_%d-%d.dat' % (self.s, self.s + self.num_data))

        if os.path.exists(file):
            with open(file, 'rb') as f:
                self.data, self.idx=pickle.load(f)
            return
            # data=self.h5load(file)
            # self.data, self.idx=data['data'], data['idx']
            # print(len(self.data), len(self.idx))
            # print(data)
            # return

        for i, p in tqdm(enumerate(paths)):
            # 读取wav文件
            wave_form, sample_freq = ta.load(p)

            wave_form = self.augment(wave_form.numpy(), sample_rate=sample_freq)

            # 计算fbank特征与其delta特征
            fbank = ta.compliance.kaldi.fbank(torch.tensor(wave_form, dtype=torch.float32), sample_frequency=sample_freq, num_mel_bins=40)
            d1 = ta.functional.compute_deltas(fbank)
            d2 = ta.functional.compute_deltas(d1)

            # 拼接在一起  (L, 39 / 120)
            feature = torch.cat([fbank, d1, d2], dim=-1)

            # Normalize acoustic feature
            feature = (feature - feature.mean(dim=0)) / (feature.std(dim=0) + 1e-10)

            self.data.append(feature)
            self.idx.append((self.s+i, feature.shape[0]))

        # self.idx.sort(key=lambda x:x[1], reverse=True)
        assert (self.s+self.num_data) > len(self.labels) or \
               len(self.data)==len(self.idx)==self.num_data, '%s %s %s'%\
                                                          (len(self.data),len(self.idx),self.num_data)

        with open(file, 'wb') as f:
            pickle.dump([self.data, self.idx], f)
        # self.h5save(file, data=self.data, idx=self.idx)



def collate_fn(dim):
    def pack(batch):
        '''
        dataloader的收集函数，用于生成一个batch。在此主要作用为padding
        :param batch: [list([feature, label])]
        :return: batch好的torch tensor
        '''
        # 计算feature与label的最大长度
        ml = max([each[0].shape[0] for each in batch])
        ml_label = max([len(each[1]) for each in batch])

        # 计算feature与label应该pad的长度
        pad_size=[ml-each[0].shape[0] for each in batch]
        lb_pad_size=[ml_label-len(each[1]) for each in batch]

        x=[]
        y=[]
        # 标记是否被pad，pad不参与loss与acc计算
        pad_map=[]
        # pad
        for i in range(len(batch)):
            x.append(torch.cat([batch[i][0], torch.zeros(pad_size[i],batch[i][0].shape[-1])],dim=dim-1))
            y.append(batch[i][1]+[1]+[2]*(lb_pad_size[i]))
            pad_map.append([1]*(len(batch[i][1])+1)+[0]*(lb_pad_size[i]))

        return torch.stack(x, dim=0), torch.tensor(y, dtype=torch.long), torch.tensor(pad_map, dtype=torch.long)

    return pack


def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn(dim=1),)




if __name__ == '__main__':
    dir=os.listdir('../datasets/ST-CMDS-20170001_1-OS/')
    wav_paths=[]
    txt_paths=[]
    for each in sorted(dir):
        if each[-4:]=='.wav':
            wav_paths.append('../datasets/ST-CMDS-20170001_1-OS/'+each)
        elif each[-4:]=='.txt':
            txt_paths.append('../datasets/ST-CMDS-20170001_1-OS/' + each)

    # print(wav_paths[:5])
    # print(txt_paths[:5])

    # count_txt(txt_paths)
    # count_feature(wav_paths)
    splits=412*240
    dataset = Data(wav_paths[:splits], txt_paths[:splits], num_data=1024)
    # dataset=Data(wav_paths[splits:], txt_paths[splits:], train=False,
    #              num_data=1024, vocab=dataset.vocab)
    #
    dl=get_dataloader(dataset, 64, False)
    print(len(dataset))
    # i=0
    for _ in range(2):
        dd = tqdm(dl)
        for x,y,z in dd:
            # print(i, end=' ')
            # i+=1
            pass
            # break
            # print(x.shape)
            # print(y.shape)
            # print(z.shape)

    # wave_form, sample_freq = ta.load(r'D:\PyPro\datasets\ST-CMDS-20170001_1-OS\20170001P00001A0001.wav')
    #
    # print(wave_form.shape)
    # print(sample_freq)
    #
    # plt.plot(range(wave_form.shape[1]), wave_form[0])
    # plt.show()
    #
    # specgram=ta.transforms.Spectrogram()(wave_form)
    # print(specgram.shape)
    # plt.imshow(specgram.log2()[0].numpy(), cmap='gray')
    # plt.show()
    #
    # mel_specgram = ta.transforms.MelSpectrogram()(wave_form)
    # print(mel_specgram.shape)
    # plt.imshow(mel_specgram.log2()[0].numpy(), cmap='gray')
    # plt.show()
    #
    # new_sample_rate=sample_freq/10
    # channel=0
    # transformed=ta.transforms.Resample(sample_freq, new_sample_rate)(wave_form[channel][None,...])
    # print(wave_form.shape, transformed.shape)
    # plt.plot(transformed[0])
    # plt.show()
    #
    # print(wave_form.min(), wave_form.max(), wave_form.mean())
    #
    #
    # mulaw=ta.transforms.MuLawEncoding()(wave_form)
    # print(mulaw.shape)
    # plt.plot(mulaw[0])
    # plt.show()
    #
    # rec = ta.transforms.MuLawDecoding()(mulaw)
    # print(rec.shape)
    # plt.plot(rec[0])
    # plt.show()
    #
    # print(((wave_form-rec).abs()/wave_form.abs()).median())
    #
    # feature1=ta.compliance.kaldi.fbank(wave_form,sample_frequency=sample_freq,num_mel_bins=40)
    # feature2=ta.compliance.kaldi.mfcc(wave_form, sample_frequency=sample_freq)
    #
    # print(feature1.shape, feature2.shape)
    # plt.imshow(feature1)
    # plt.show()
    # plt.imshow(feature2)
    # plt.show()
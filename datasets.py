import torchaudio as ta
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
from memory_profiler import profile
import pickle

def count_txt(txt_paths):
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
                del line
    print(f'最长句子长度：{mx}, 最短句子长度：{mn}, 平均句子长度：{avg/c}')
    plt.bar(d.keys(), d.values())
    plt.show()


def count_feature(wav_paths):
    mn=1000000
    mx=-1
    avg=0
    c=0
    d={}
    for file in tqdm(wav_paths):
        wave_form, sample_freq=ta.load(file)
        # length=ta.compliance.kaldi.mfcc(wave_form, sample_frequency=sample_freq).shape[0]
        length=wave_form.shape[1]//160
        mx=max(mx, length)
        d[length]=d.get(length, 0)+1
        mn = min(mn, length)
        avg+=length
        c+=1
        del wave_form, sample_freq, length
    print(f'最长语音特征长度：{mx}, 最短语音特征长度：{mn}, 平均语音特征长度：{avg/c}')
    plt.bar(d.keys(), d.values())
    plt.show()

class Data(Dataset):
    def __init__(self, wav_paths, txt_paths, train=True, vocab=None, save_path='./words_vocab_label.pkl'):
        super().__init__()

        assert train or vocab is not None, 'validation时需要传入vocab'

        self.wav_paths=wav_paths
        self.txt_paths=txt_paths

        if train and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.words, self.vocab, self.labels=pickle.load(f)
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
                    pickle.dump([self.words, self.vocab, self.labels], f)
        assert len(self.labels)==len(self.wav_paths), '%s %s'%(len(self.labels), len(self.wav_paths))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wave_form, sample_freq = ta.load(self.wav_paths[idx])

        MFCC = ta.compliance.kaldi.mfcc(wave_form, sample_frequency=sample_freq)
        d1 = ta.functional.compute_deltas(MFCC)
        d2 = ta.functional.compute_deltas(d1)

        feature = torch.cat([MFCC, d1, d2], dim=-1)

        feature=(feature-feature.mean())/(feature.std()+1e-10)

        return feature, self.labels[idx]

def collate_fn(dim):
    def pack(batch):
        ml = max([each[0].shape[0] for each in batch])
        ml_label = max([len(each[1]) for each in batch])

        pad_size=[ml-each[0].shape[0] for each in batch]
        lb_pad_size=[ml_label-len(each[1]) for each in batch]

        x=[]
        y=[]
        pad_map=[]
        for i in range(len(batch)):
            # print(len(pad_size), i)
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

    print(wav_paths[:5])
    print(txt_paths[:5])

    # count_txt(txt_paths)
    # count_feature(wav_paths)
    # dataset=Data(wav_paths[:10], txt_paths[:10])

    # dl=get_dataloader(dataset, 2, False)
    # print(len(dataset))
    # for x,y,z in dl:
    #     print(x.shape)
    #     print(y.shape)
    #     print(z.shape)

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
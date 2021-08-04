from pyaudio import PyAudio, paInt16
import numpy as np
import wave
from models import *
import torchaudio as ta
import pickle


def recode(second):
    p=PyAudio()

    s1 = p.open(format=paInt16, channels=2, rate=16000, output=True)
    s2 = p.open(format=paInt16, channels=2, rate=16000, input=True)

    print("START")
    data=s2.read(second*16000)
    print('END')
    s1.write(data)

    array=np.frombuffer(data, dtype=np.dtype('<i2'))
    array = torch.tensor(array)[None, ...].float()

    array=(array-array.mean())/(array.std()+1e-10)

    feature=ta.compliance.kaldi.fbank(array, sample_frequency=16000, channel=1)
    delta1=ta.functional.compute_deltas(feature)
    delta2=ta.functional.compute_deltas(delta1)

    feature=torch.cat([feature, delta1, delta2], dim=-1)

    # feature = (feature-feature.mean(dim=0))/(feature.std(dim=0)+1e-10)
    feature = (feature - feature.mean()) / (feature.std() + 1e-10)

    return array,feature



if __name__ == "__main__":
    model = torch.load('./models/loss_1.0261238430906152_val_loss_1.4254556577882649.pth')
    model.to('cuda')
    # with open('./vocab_label.pkl', 'rb') as f:
    #     vocab, _=pickle.load(f)
    # words=list(vocab.keys())
    # c, d=recode(3)
    w=wave.open(r'D:\PyPro\datasets\ST-CMDS-20170001_1-OS\20170001P00001A0001.wav', 'rb')
    print(w.getnchannels())
    print(w.getnframes())
    print(w.getframerate())
    # feature = ta.compliance.kaldi.mfcc(d, sample_frequency=f)
    # delta1 = ta.functional.compute_deltas(feature)
    # delta2 = ta.functional.compute_deltas(delta1)
    #
    # feature = torch.cat([feature, delta1, delta2], dim=-1)
    #
    # feature = (feature - feature.mean()) / (feature.std() + 1e-10)
    #
    # ans=model.predict_(feature.unsqueeze(0).to('cuda'))
    # print(' '.join([words[i] for i in ans]))

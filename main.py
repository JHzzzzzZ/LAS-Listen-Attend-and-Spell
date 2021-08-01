from datasets import *
from models import *
from pytorch_lamb import Lamb

device='cuda'
if __name__ == '__main__':
    dir = os.listdir('../datasets/ST-CMDS-20170001_1-OS/')
    wav_paths = []
    txt_paths = []
    for each in sorted(dir):
        if each[-4:] == '.wav':
            wav_paths.append('../datasets/ST-CMDS-20170001_1-OS/' + each)
        elif each[-4:] == '.txt':
            txt_paths.append('../datasets/ST-CMDS-20170001_1-OS/' + each)

    length=len(wav_paths)
    splits=int(0.9*length)
    dataset=Data(wav_paths[:splits], txt_paths[:splits], train=True)
    val_dataset = Data(wav_paths[splits:], txt_paths[splits:], train=False, vocab=dataset.vocab)

    dl=get_dataloader(dataset, 64, True)
    val_dl = get_dataloader(val_dataset, 128, False)

    model = Model(39, 512, len(dataset.vocab)).to(device)
    optimizer = Lamb(model.parameters(), lr=0.01, weight_decay=5e-3)
    schedule=optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, eta_min=1e-4, T_mult=2)

    epochs=50
    length=len(dl)

    for epoch in range(epochs):
        ls=0
        accs=0
        val_ls=0
        val_accs=0
        model.train()
        dd=tqdm(dl)
        for i,(x,y,flag) in enumerate(dd):
            x=x.to(device)
            y=y.to(device)
            flag=flag.to(device)
            c=0
            y_, loss = model(x,y,flag)

            acc = model.mask_acc(y_, y, flag)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            schedule.step(epoch+i/length)

            l_temp=loss.detach().cpu().item()
            acc_temp=acc.detach().cpu().item()
            # print(f'step:{c},loss:{l_temp},acc:{acc_temp}, lr:{optimizer.param_groups[0]["lr"]}')
            c+=1
            dd.set_description(f'loss:{l_temp},acc:{acc_temp},lr:{optimizer.param_groups[0]["lr"]}')
                    # print(loss.detach().cpu().item(), acc.detach().cpu().item())
            ls+=l_temp
            accs+=acc_temp

        model.eval()
        tt=tqdm(val_dl)
        for x,y,flag in tt:
            x = x.to(device)
            y = y.to(device)
            flag = flag.to(device)

            with torch.no_grad():

                y_, loss = model(x,y,flag)

                acc = model.mask_acc(y_, y, flag)

            l_temp = loss.detach().cpu().item()
            acc_temp = acc.detach().cpu().item()

            tt.set_description(f'val_loss:{l_temp},val_acc:{acc_temp}')

            val_ls += loss.detach().cpu().item()
            val_accs += acc.detach().cpu().item()

        print(
            f'Epoch {epoch + 1}:\n\tloss:{ls / len(dl)}, acc:{accs / len(dl)}, val_loss:{val_ls / len(val_dl)}, val_acc:{val_accs / len(val_dl)}')
        torch.save(model, f'./models/loss_{ls/len(dl)}_val_loss_{val_ls/len(val_dl)}.pth')



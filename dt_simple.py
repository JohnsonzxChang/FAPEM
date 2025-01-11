import torch
import numpy as np
import scipy.io as mio
import torch.utils.data as DD
from scipy.signal import kaiserord, filtfilt, firwin, butter
from conf import Config

# BENCH_DATASET_DIR = '/Users/zhangxu/Desktop/ssvep/bench/'
BENCH_DATASET_DIR = '/home/zhangxu/prj/ssvep_project/bench'

def construct_filter(fs=250.0, cutoff_hz=[[6, 66]], type='kaiser'):
    if type == 'kaiser':
        width = 4.0 / fs
        ripple_db = 20.0
        N, beta = kaiserord(ripple_db, width)
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(
                firwin(N, [cutoff_hz[i][0], cutoff_hz[i][1]], window=('kaiser', beta), fs=fs, pass_zero="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j], 1.0, x, axis=-1, padlen=x.shape[-1] - 2) for j in
                               range(len(cutoff_hz))]
    elif type == 'firwin':
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(firwin(25, [cutoff_hz[i][0], cutoff_hz[i][1]], fs=fs, pass_zero="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j], 1.0, x, axis=-1, padlen=x.shape[-1] - 2) for j in
                               range(len(cutoff_hz))]
    elif type == 'butter':
        taps = []
        for i in range(len(cutoff_hz)):
            taps.append(butter(4, [cutoff_hz[i][0], cutoff_hz[i][1]], fs=fs, btype="bandpass"))
        my_filter = lambda x: [filtfilt(taps[j][0], taps[j][1], x, axis=-1, padlen=x.shape[-1] - 2) for j in
                               range(len(cutoff_hz))]
    else:
        raise ValueError('type err')
    return my_filter

def load_data_refine(base_path: str, T: int, E: list, Mm=None, T0: int = 125, id_0=0, filterbank=None):
    assert id_0 in [0, 70], f'Id0 wrong {id_0}'
    if Mm is None:
        Mm = [53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64]
    f0 = f'{base_path}/Freq_Phase.mat'
    Freq_Class = mio.loadmat(f0)['freqs'].squeeze()
    Phase_Class = mio.loadmat(f0)['phases'].squeeze()
    res = {}
    res["Freq_Class"] = np.round(Freq_Class, 2)
    res["Phase_Class"] = np.round(Phase_Class, 2)
    tmp = np.load(f'{base_path}/ssvep_dataset_data.npy')[E, :, :]
    tmp = tmp[:, Mm, :]
    lb = np.load(f'{base_path}/ssvep_dataset_label.npy')[E]

    # pre 125, valid 1250, post 1375, finish 1500, cut from 250 to 1250 (4s)
    if filterbank is not None:
        tmp = np.stack(filterbank(tmp), axis=1)
        assert tmp.shape[1] == 3
        print('finish filter...')
    print(f'loading data from {T0} to {T0 + T}')
    tmp = tmp[:, :, :, T0:T0 + T]
    res["Data"] = tmp
    res["Label"] = lb
    res["id"] = [(e // 240 + id_0) for e in E]
    if id_0 == 0:
        assert max(res["id"]) <= 35
    else:
        assert max(res["id"]) <= 105
    return res

class base_torch_mix_dataset(DD.Dataset):
    def __init__(self, T0: int, EE_Train=None, EE_Test=None, deltT=10,
                 preprocess=False, multiplex=1, use_beta=False, M=None,
                 filterbank=False, t_pre=125, train=True):
        super().__init__()
        self.T0 = T0
        self.multiplex = multiplex
        self.base_path = BENCH_DATASET_DIR
        self.aux_t = deltT
        self.use_beta = use_beta
        assert self.use_beta == False
        self.M = M
        self.EE_Train = EE_Train
        self.EE_Test = EE_Test

        self.preprocess = preprocess
        if filterbank != False:
            print('construct filter bank')
            assert type(filterbank) == int
            filterbank = [[(i + 1) * 8 - 2, 90] for i in range(filterbank)]
            self.filterbank = construct_filter(cutoff_hz=filterbank, type='butter')
        else:
            print('dont use filter bank')
            self.filterbank = None

        if train is None:
            assert self.EE_Train is not None and self.EE_Test is not None
            self.E = self.EE_Train + self.EE_Test
            raise ValueError('train is None')
        elif train == True:
            self.E = self.EE_Train
        elif train == False:
            self.E = self.EE_Test
        else:
            raise ValueError(f'self E value err with {train}')
        
        assert t_pre == 125+35, f't_pre {t_pre}, should be 125+35'
        self.t_pre = t_pre
        self.ad_label = np.ones((250*6,), dtype=float)
        self.ad_label[0:125] = 0
        for k in range(35):
            self.ad_label[125+k] = k/35
        self._load_data()
        self._make_label()


    def _load_data(self):
        print(f'cutting data into {self.multiplex} parts')
        assert int(self.T0 * (1+self.multiplex)) <= 5 * 250, f'{self.T0} * {self.multiplex} vs 1250'
        res = load_data_refine(self.base_path, int(self.T0 * (1+self.multiplex) + self.t_pre), self.E, T0=0, Mm=self.M, filterbank=self.filterbank)
        self.data = res["Data"][:,:,:,self.t_pre:self.t_pre+int(self.T0*self.multiplex)]
        self.data_aux = res["Data"][:,:,:,self.t_pre+self.aux_t:self.t_pre+int(self.T0+self.aux_t*(1+self.multiplex))]
        self.ad_t0 = np.random.randint(125-self.T0, self.t_pre+self.T0, size=(self.data_aux.shape[0]*self.multiplex,), dtype=int)
        self.data_ad = res["Data"]
        self.freq = res["Label"]
        self.Freq_Class = res["Freq_Class"]
        self.id = np.array(res["id"])

    def _make_label(self):
        assert self.data.shape[0] == self.freq.shape[0]
        self.label = np.zeros_like(self.freq, dtype=int)
        for i in range(self.label.shape[0]):
            tmp = np.where(np.round(self.Freq_Class, 2) == np.round(self.freq[i],2))
            assert len(tmp) == 1
            self.label[i] = int(tmp[0])
        data = []
        data_aux = []
        label = []
        idid = []
        for i in range(self.multiplex):
            data.append(self.data[:, :, :, i * self.T0:(i + 1) * self.T0])
            data_aux.append(self.data_aux[:, :, :, i * self.aux_t:i * self.aux_t + self.T0])
            label.append(self.label)
            idid.append(self.id)
        self.data = np.concatenate(data, axis=0)
        self.data_aux = np.concatenate(data_aux, axis=0)
        self.label = np.concatenate(label, axis=0)
        self.id = np.concatenate(idid, axis=0)
        data_ad = []
        label_ad = []
        for j in range(self.multiplex):
            for i in range(self.data_ad.shape[0]):
                data_ad.append(self.data_ad[i, :, :, self.ad_t0[i+j*self.data_ad.shape[0]]:self.ad_t0[i+j*self.data_ad.shape[0]]+self.T0])
                label_ad.append(self.ad_label[self.ad_t0[i+j*self.data_ad.shape[0]]:self.ad_t0[i+j*self.data_ad.shape[0]]+self.T0])
                assert data_ad[-1].shape == (3, 9, self.T0), f'{data_ad[-1].shape} @ {i}, {j} @ {self.data_ad.shape}, {self.ad_t0[i+j*self.data_ad.shape[0]]}'
        self.data_ad = np.array(data_ad)
        self.label_ad = np.array(label_ad)
        print("ssssssssssssssssssssssssssssssssssssssss", self.id.shape, self.label.shape, self.data.shape, self.data_aux.shape, self.data_ad.shape, self.label_ad.shape)

    def _preprocess_data(self, x):
        x = x - np.mean(x, axis=-1, keepdims=True)
        x = x / np.amax(x, axis=-1, keepdims=True)
        return x

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        data = self.data[item]
        data_aux = self.data_aux[item]
        data_ad = self.data_ad[item]
        if self.preprocess:
            data = self._preprocess_data(data)
            data_aux = self._preprocess_data(data_aux)
            data_ad = self._preprocess_data(data_ad)
        data = torch.tensor(data).type(torch.float)
        data_aux = torch.tensor(data_aux).type(torch.float)
        data_ad = torch.tensor(data_ad).type(torch.float)
        id = torch.tensor(self.id[item]).type(torch.LongTensor)
        label = torch.tensor(self.label[item]).type(torch.LongTensor)
        label_ad = torch.tensor(self.label_ad[item]).type(torch.float)
        return {
            "data": data, 
            "id": id, 
            "label": label, 
            "data_aux": data_aux,
            "data_ad": data_ad,
            "label_ad": label_ad
            }
     
        
def get_dataloader(config):
    assert isinstance(config, Config)
    my_t0 = config.data_t0
    my_channel = config.channel
    my_worker = config.worker
    batch_size = config.batch_size
    use_beta = config.dataset == 'beta'
    val_idx = config.idx
    val_person_id = config.val_person_id
    init_t0 = config.init_t0
    preprocess = True
    multiplex = config.multiplex
    EE_Train = []
    EE_Test = []
    trn_person_id = list(range(70)) if use_beta else list(range(35))
    session = 4 if use_beta else 6
    assert val_idx < session
    if val_person_id is not None:
        trn_person_id.remove(val_person_id)
    for e in trn_person_id:
        rg = list(range(e * session * 40, e * session * 40 + session * 40))
        rg_val = list(range(e * session * 40 + val_idx * 40, e * session * 40 + (val_idx + 1) * 40))
        rg = list(set(rg) - set(rg_val))
        EE_Train += rg
        EE_Test += rg_val

    print(len(EE_Train),len(EE_Test))
    
    a = base_torch_mix_dataset(T0=my_t0, EE_Train=EE_Train, EE_Test=EE_Test, deltT=config.delt_t, preprocess=preprocess, multiplex=multiplex, use_beta=use_beta, M=my_channel, filterbank=3, train=True, t_pre=init_t0)

    b = base_torch_mix_dataset(T0=my_t0, EE_Train=EE_Train, EE_Test=EE_Test, deltT=config.delt_t, preprocess=preprocess, multiplex=multiplex, use_beta=use_beta, M=my_channel, filterbank=3, train=False, t_pre=init_t0)
    
    
    return {
        'trn_dataloader': DD.DataLoader(a,
                                     batch_size=batch_size, shuffle=True, pin_memory=True,
                                     num_workers=my_worker),
        'val_dataloader': DD.DataLoader(b,
                                     batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=my_worker)
    }


if __name__ == "__main__":
    my_dataset = get_dataloader(config=Config())
    
    cnt = 0
    for i, batch in enumerate(my_dataset["trn_dataloader"]):
            x = batch["data"].squeeze(0)
            id = batch["id"].squeeze(0)
            y = batch["label"].squeeze(0)
            cnt += y.shape[0]
    print(cnt/35)
    
    cnt = 0
    for i, batch in enumerate(my_dataset["val_dataloader"]):
            x = batch["data"].squeeze(0)
            id = batch["id"].squeeze(0)
            y = batch["label"].squeeze(0)
            cnt += y.shape[0]
    print(cnt/35)
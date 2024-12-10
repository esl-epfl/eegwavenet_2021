import numpy as np
import torch
import torch.nn as nn
from scipy import signal
import os
from thuwajit_2021.architecture import EEGWaveNet

def load_models(device):
    models = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path, 'models')
    for fold in range(5):
        Model = EEGWaveNet(18, 2).float()
        Model.load_state_dict(torch.load(os.path.join(models_path, 'model_fold_{}.pt'.format(fold+1)), weights_only=True))
        Model.eval()
        Model.to(device)
        models.append(Model)
    return models

class SeizureDataset(nn.Module):
    def __init__(self, data, window_size_sec, fs):
        super(SeizureDataset, self).__init__()

        preprocessed_data = self.preprocess(data, fs, cutoff=64)   # low-pass filter at 64 Hz
        self.data = preprocessed_data
        self.window_size = int(window_size_sec*fs)
        self.fs = fs
        self.recording_duration = int(data.shape[1] / fs)

        window_idx = np.arange(0, self.recording_duration*self.fs, self.fs).astype(int)
        self.window_idx = window_idx[window_idx < self.recording_duration*self.fs - self.window_size]

    def __len__(self):
        return len(self.window_idx)
    
    def preprocess(self, data, fs, cutoff=64):
        b, a = signal.butter(4, 64, fs=fs, btype='low', analog=False)   # low-pass filter at 64 Hz
        data_filtered = np.zeros_like(data)
        for channel in range(data.shape[0]):
            data_filtered[channel, :] = signal.filtfilt(b, a, data[channel, :])
        return data_filtered
    
    def __getitem__(self, idx):
        eeg_clip = self.data[:, self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        return torch.tensor(eeg_clip)
    
def get_dataloader(data, window_size_sec, fs, batch_size):
    dataset = SeizureDataset(data, window_size_sec, fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def predict(models, dataloader, device, recording_duration, window_size_sec=4, overlap_size_sec=3, fs=256):
    window_size = int(window_size_sec*fs)
    overlap_size = int(overlap_size_sec*fs)
    models_detections = np.zeros((len(models), int(recording_duration*fs)))
    y_predict = np.zeros(int(recording_duration*fs))
    for k, model in enumerate(models):
        y_preds = []
        with torch.no_grad():
            for data in dataloader:
                data = data.float().to(device)
                pred = model(data)
                pred = list(np.argmax(list(pred.cpu().detach().numpy()), axis=1))
                y_preds += pred
        y_preds = np.array(y_preds)

        for i in range(len(y_preds)):
            # for each time point, assign majority of all overlaps
            overlap_left = max(0, i*(window_size-overlap_size))
            overlap_right = min(recording_duration*fs, (i+1)*(window_size-overlap_size))
            majority_vote = np.mean(y_preds[max(0, i-int(overlap_size_sec//2)):
                                            min(i+int(overlap_size_sec//2), recording_duration)]) > 0.5
            models_detections[k, int(overlap_left):int(overlap_right)] = majority_vote.astype(int)
    y_predict = (np.mean(models_detections, axis=0) > 0.5).astype(int)
    return y_predict


import torch
import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from thuwajit_2021.utils import load_models, get_dataloader, predict

def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)

    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar() # re-reference to bipolar montage

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fs = eeg.fs
    window_size_sec = 4
    overlap_size_sec = 3
    recording_duration = eeg.data.shape[1] / fs

    models = load_models(device)
    dataloader = get_dataloader(eeg.data, window_size_sec, fs, batch_size=512)
    y_predict = predict(models, dataloader, device, recording_duration, 
                        window_size_sec=window_size_sec, overlap_size_sec=overlap_size_sec, fs=fs)

    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)



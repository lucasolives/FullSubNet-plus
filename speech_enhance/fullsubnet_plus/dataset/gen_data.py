import glob
import random
import os
from pathlib import Path

import tqdm
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from dataset_train import Dataset
from audiomentations_augment import NoiseAugmentation

if __name__ == "__main__":
    NUM_AUDIOS = 10
    SAMPLE_RATE = 16000
    output_dir = Path("test_fullsubnet_1/")
    os.makedirs(str(output_dir), exist_ok=True)
    audios = glob.glob("/home/lucas/Documents/alcateia/FullSubNet-plus/speech_enhance/fullsubnet_plus/dataset/dev-clean/**/*.wav", recursive=True)
    random.seed(727)
    selected_audios = random.sample(audios, NUM_AUDIOS)
    #selected_audios_paths = ','.join(selected_audios)
    augmenter = NoiseAugmentation(sr=SAMPLE_RATE)
    #print(selected_audios_paths)
    #print(audios)
    # augmenter = Dataset(clean_dataset=selected_audios_paths, 
    #                     sr=SAMPLE_RATE, 
    #                     pre_load_clean_dataset=True,
    #                     num_workers=4,)
    for audio_path in tqdm.tqdm(selected_audios):
        name = Path(audio_path).name
        stem = Path(audio_path).stem
        suffix = Path(audio_path).suffix
        audio, sr = torchaudio.load(audio_path)
        resampler = T.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)
        aug_audio = augmenter.augment(audio.detach().cpu().numpy(), sample_rate=SAMPLE_RATE)
        aug_audio = torch.tensor(aug_audio)
        torchaudio.save(str(output_dir / f"{stem}_aug.{suffix}"), aug_audio, SAMPLE_RATE)
        torchaudio.save(str(output_dir / name), audio, SAMPLE_RATE)


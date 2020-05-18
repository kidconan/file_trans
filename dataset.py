import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio as Audio
from text import text_to_sequence
from utils import process_text, pad_1D, pad_2D
from pad_skill import _pad_mel, _prepare_data
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self, csv_file, root_dir):
        # self.text = process_text(os.path.join("data", "train.txt"))
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        # return len(self.text)
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0][2:]) + '.wav'
        text = self.landmarks_frame.ix[idx, 1]
        alignment = np.load(hparams.alignment_path + '/' + str(idx) + '.npy')

        text = np.asarray(text_to_sequence(text, hparams.text_cleaners), dtype=np.int32)
        mel = np.load(wav_name[:-4] + '.pt.npy')
        # mel_input = np.concatenate([np.zeros([1,hparams.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'pos_mel':pos_mel, 'pos_text':pos_text, 'alignment':alignment}

        return sample
        # mel_gt_name = os.path.join(
        #     hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (idx+1))
        # mel_gt_target = np.load(mel_gt_name)
        # D = np.load(os.path.join(hparams.alignment_path, str(idx)+".npy"))

        # character = self.text[idx][0:len(self.text[idx])-1]
        # character = np.array(text_to_sequence(
        #     character, hparams.text_cleaners))

        # sample = {"text": character,
        #           "mel_target": mel_gt_target,
        #           "D": D}

        return sample


def reprocess(batch, cut_list):

    text = [batch[ind]["text"] for ind in cut_list]
    mel = [batch[ind]["mel"] for ind in cut_list]
    # mel_input = [batch[ind]['mel_input'] for ind in cut_list]
    text_length = [batch[ind]['text_length'] for ind in cut_list]
    pos_mel = [batch[ind]['pos_mel'] for ind in cut_list]
    pos_text= [batch[ind]['pos_text'] for ind in cut_list]
    alignment = [batch[ind]['alignment'] for ind in cut_list]

    mel_max_len = max(list(map(lambda x: x.shape[0], mel)))
    
    text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
    mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
    alignment = [i for i, _ in sorted(zip(alignment, text_length), key=lambda x: x[1], reverse=True)]
    # mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
    pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
    pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
    text_length = sorted(text_length, reverse=True)
    # PAD sequences with largest length of the batch
    text = _prepare_data(text).astype(np.int32)
    mel = _pad_mel(mel)
    # mel_input = _pad_mel(mel_input)
    pos_mel = _prepare_data(pos_mel).astype(np.int32)
    pos_text = _prepare_data(pos_text).astype(np.int32)
    alignment = _prepare_data(alignment).astype(np.int32)

    out = {"text": text,
           "mel_target": mel,
           "D": alignment,
           "mel_pos": pos_mel,
           "src_pos": pos_text,
           # "mel_max_len": int(max(text_length))}
           "mel_max_len": mel_max_len}
    
    # texts = [batch[ind]["text"] for ind in cut_list]
    # mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    # Ds = [batch[ind]["D"] for ind in cut_list]

    # length_text = np.array([])
    # for text in texts:
    #     length_text = np.append(length_text, text.shape[0])

    # src_pos = list()
    # max_len = int(max(length_text))
    # for length_src_row in length_text:
    #     src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
    #                           (0, max_len-int(length_src_row)), 'constant'))
    # src_pos = np.array(src_pos)

    # length_mel = np.array(list())
    # for mel in mel_targets:
    #     length_mel = np.append(length_mel, mel.shape[0])

    # mel_pos = list()
    # max_mel_len = int(max(length_mel))
    # for length_mel_row in length_mel:
    #     mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
    #                           (0, max_mel_len-int(length_mel_row)), 'constant'))
    # mel_pos = np.array(mel_pos)

    # texts = pad_1D(texts)
    # Ds = pad_1D(Ds)
    # mel_targets = pad_2D(mel_targets)



    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # Test
    dataset = FastSpeechDataset()
    training_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            # print(mel_target.size())
            # print(D.sum())
            print(cnt)
            if mel_target.size(1) == D.sum().item():
                cnt += 1

    print(cnt)

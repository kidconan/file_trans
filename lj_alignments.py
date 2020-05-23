import os
from torch.utils.data import Dataset
from preprocess import DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import librosa
import pandas as pd
from text import text_to_sequence


class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0][2:]) + '.wav'
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        text = self.landmarks_frame.ix[idx, 1]

        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample

def get_dataset():
    return LJDatasets(os.path.join('./data/LJSpeech-1.1','metadata.csv'), os.path.join('./data/LJSpeech-1.1', 'wavs'))


def get_D(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[1])])

    for i in range(np.shape(alignment)[0]):
        max_index = alignment[i].tolist().index(alignment[i].max())
        D[max_index] = D[max_index] + 1

    return D

if not os.path.exists('alignments'):
    os.mkdir('alignments')
check_point = './checkpoint/checkpoint_transformer_820000.pth.tar'
para_file = t.load(check_point, map_location={'cuda:5':'cuda:0'})

model = nn.DataParallel(Model().cuda())
model.load_state_dict(para_file['model'])
model.eval()
for epoch in range(1):

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=False, num_workers=1)
    k = 0
    # pbar = tqdm(dataloader)
    # for i, data in enumerate(pbar):
    for character, mel, mel_input, pos_text, pos_mel, _  in dataloader:
        # pbar.set_description("Processing at epoch %d"%epoch)
            
        # character, mel, mel_input, pos_text, pos_mel, _ = data
        
        stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
        
        character = character.cuda()
        mel = mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        pos_mel = pos_mel.cuda()
        
        _, _, attn_probs, _, _, _ = model.forward(character, mel_input, pos_text, pos_mel)

        attn_probs = attn_probs[-1].sum(dim=0)
        attn_probs = attn_probs.data.cpu().numpy()
        # attn_probs = attn_probs.data.cpu().argmax(dim=-1)
        # attn_probs = attn_probs.numpy()
        # print(attn_probs.shape)
        attn_probs = get_D(attn_probs)
        np.save('./alignments/'+str(k)+'.npy', attn_probs)
        k += 1
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7,8'

check_point = ''
para_file = t.load(check_point)
model = Model().cuda()
model.eval()
for epoch in range(hp.epochs):

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        pbar.set_description("Processing at epoch %d"%epoch)
            
        character, mel, mel_input, pos_text, pos_mel, _ = data
        
        stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
        
        character = character.cuda()
        mel = mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        pos_mel = pos_mel.cuda()
        
        _, _, _, _, _, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)

        attns_dec = attns_dec[0]
        attns_dec = attns_dec.data.cpu().tolist()
        print(attns_dec)
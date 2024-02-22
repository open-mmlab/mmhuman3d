""" Use a trained betas adapter to fix params. Must fix global orient transl first. """
import os.path as osp
import os
import glob
import tqdm
import torch
import numpy as np
import torch.nn as nn

import pdb

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class BetasAdapter(nn.Module):
    def __init__(self):
        super(BetasAdapter, self).__init__()
        self.fc1 = nn.Linear(10, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 10)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def fix_humandata(load_path):
    human_data = np.load(load_path, allow_pickle=True)
    gender = human_data['meta'].item()['gender']
    smplx = human_data['smplx'].item()
    betas = smplx['betas']

    new_betas = []
    assert len(gender) == len(betas)
    for gen, bet in tqdm.tqdm(zip(gender, betas), total=len(gender)):
        assert gen in ('male', 'female', 'neutral'), f'gen: {gen}'

        with torch.no_grad():
            if gen == 'male':
                new_bet = smplx_male_to_smplx_neutral(torch.tensor(bet.reshape(1, 10), device=device))
                new_bet = new_bet.detach().cpu().numpy().reshape(10)
                assert not np.allclose(bet, new_bet)
            elif gen == 'female':
                new_bet = smplx_female_to_smplx_neutral(torch.tensor(bet.reshape(1, 10), device=device))
                new_bet = new_bet.detach().cpu().numpy().reshape(10)
                assert not np.allclose(bet, new_bet)
            else:
                new_bet = bet.copy()

        new_betas.append(new_bet)

    new_betas = np.stack(new_betas, axis=0)
    assert new_betas.shape == betas.shape

    new_smplx = { k: v for k, v in smplx.items() }
    new_smplx['betas_neutral'] = new_betas

    new_human_data = {}
    for k, v in human_data.items():
        if len(v.shape) == 0:
            new_human_data[k] = v.item()
        else:
            new_human_data[k] = v
    new_human_data['smplx'] = new_smplx

    stem, _ = osp.splitext(osp.basename(load_path))
    save_stem = stem + '_fix_betas'
    save_path = load_path.replace(stem, save_stem)
    np.savez_compressed(save_path, **new_human_data)
    print(load_path, '->', save_path)


if __name__ == '__main__':
    base_path = 'data/body_models/convert'
    smplx_female_to_smplx_neutral_path = osp.join(base_path, 'smplx_female_to_smplx_neutral.pth')
    smplx_male_to_smplx_neutral_path = osp.join(base_path, 'smplx_male_to_smplx_neutral.pth')

    smplx_male_to_smplx_neutral = BetasAdapter()
    smplx_male_to_smplx_neutral.load_state_dict(torch.load(smplx_male_to_smplx_neutral_path, map_location=device))
    smplx_male_to_smplx_neutral.to(device)

    smplx_female_to_smplx_neutral = BetasAdapter()
    smplx_female_to_smplx_neutral.load_state_dict(torch.load(smplx_female_to_smplx_neutral_path, map_location=device))
    smplx_female_to_smplx_neutral.to(device)

    work_dir = '/mnt/d/datasets/moyo/output'

    # # egobody
    # load_paths = sorted(glob.glob(osp.join(work_dir, 'egobody_*.npz')))
    # load_paths = [p for p in load_paths if 'fix_betas' not in p]
    # for load_path in load_paths:
    #     fix_humandata(load_path)

    # # synbody
    # load_paths = [osp.join(work_dir, 'synbody_train_230521_04000.npz')]
    # load_paths = [p for p in load_paths if 'fix_betas' not in p]
    # for load_path in load_paths:
    #     fix_humandata(load_path)

    # renbody
    load_paths = sorted(glob.glob(osp.join(work_dir, '*.npz')))
    load_paths = [p for p in load_paths if 'fix_betas' not in p]
    for load_path in load_paths:
        fix_humandata(load_path)

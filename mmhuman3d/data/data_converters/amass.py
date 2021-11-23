import glob
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS

all_sequences = [
    'ACCAD',
    'BioMotionLab_NTroje',
    'CMU',
    'EKUT',
    'Eyes_Japan_Dataset',
    'HumanEva',
    'KIT',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SSM_synced',
    'TCD_handMocap',
    'TotalCapture',
    'Transitions_mocap',
]


@DATA_CONVERTERS.register_module()
class AmassConverter(BaseConverter):
    """AMASS dataset
    `AMASS: Archive of Motion Capture as Surface Shapes' ICCV`2019
    More details can be found in the `paper
    <https://files.is.tue.mpg.de/black/papers/amass.pdf>`__.
    """

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys video_path, smplh, meta, frame_idx
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        video_path_, frame_idx_ = [], []
        smplh = {}
        smplh['body_pose'] = []
        smplh['global_orient'] = []
        smplh['betas'] = []
        smplh['transl'] = []
        smplh['left_hand_pose'] = []
        smplh['right_hand_pose'] = []
        meta = {}
        meta['gender'] = []
        annot_dir = dataset_path

        for seq_name in tqdm(all_sequences):
            seq_folder = os.path.join(annot_dir, seq_name)

            subjects = os.listdir(seq_folder)
            for subject in tqdm(subjects):
                pattern = os.path.join(seq_folder, subject, '*.npz')
                action_list = sorted(glob.glob(pattern))

                for action_file in action_list:
                    if action_file.endswith('shape.npz'):
                        continue

                    data = np.load(action_file)

                    # get smpl data
                    gender = data['gender']
                    betas = data['betas'][:10].reshape((-1, 10))
                    trans = data['trans'].reshape((-1, 3))
                    root_orient = data['poses'][:, :3]
                    pose_body = data['poses'][:, 3:66].reshape((-1, 21, 3))
                    pose_hand = data['poses'][:, 66:]
                    left_hand_pose = pose_hand[:, :45].reshape(-1, 15, 3)
                    right_hand_pose = pose_hand[:, 45:].reshape(-1, 15, 3)

                    # get video file
                    action_name = action_file.split('/')[-1].split('_poses')[0]
                    vid_id = os.path.join(seq_name, subject,
                                          action_name + '.mp4')
                    mocap_framerate = int(data['mocap_framerate'])
                    sampling_freq = mocap_framerate // 10

                    num_frames = pose_body.shape[0]

                    for i in range(num_frames):
                        if i % sampling_freq != 0:
                            continue
                        smplh['body_pose'].append(pose_body[i])
                        smplh['global_orient'].append(root_orient[i])
                        smplh['betas'].append(betas)
                        smplh['transl'].append(trans[i])
                        smplh['left_hand_pose'].append(left_hand_pose[i])
                        smplh['right_hand_pose'].append(right_hand_pose[i])
                        meta['gender'].append(gender)
                        video_path_.append(vid_id)
                        frame_idx_.append(i)

        # change list to np array
        smplh['body_pose'] = np.array(smplh['body_pose']).reshape((-1, 21, 3))
        smplh['global_orient'] = np.array(smplh['global_orient']).reshape(
            (-1, 3))
        smplh['betas'] = np.array(smplh['betas']).reshape((-1, 10))
        smplh['transl'] = np.array(smplh['transl']).reshape((-1, 3))
        smplh['left_hand_pose'] = np.array(smplh['left_hand_pose']).reshape(
            (-1, 15, 3))
        smplh['right_hand_pose'] = np.array(smplh['right_hand_pose']).reshape(
            (-1, 15, 3))
        meta['gender'] = np.array(meta['gender'])

        human_data['video_path'] = video_path_
        human_data['frame_idx'] = np.array(frame_idx_).reshape(-1)
        human_data['meta'] = meta
        human_data['config'] = 'amass'
        human_data['smplh'] = smplh

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'amass.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)

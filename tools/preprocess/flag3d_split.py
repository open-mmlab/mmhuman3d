import cv2
import os
import argparse
import glob
import pdb
import joblib

from tqdm import tqdm


def reformat_flag3d(args):

    kp_p = os.path.join(args.flag3d_path, 'flag3d_keypoint.pkl')
    with open(kp_p, 'rb') as f:
        kp_param = joblib.load(f)

    annos = {}
    for annotation in kp_param['annotations']:
        anno = annotation
        annos[annotation['frame_dir']] = annotation
        del anno['frame_dir']


    kp_param['annotations'] = annos

    kp_pr = kp_p.replace('.pkl', '_reformat.pkl')
    with open(kp_pr, 'wb') as f:
        joblib.dump(kp_param, f)


def split_flag3d(args):

    # parse sequences
    videos = glob.glob(os.path.join(args.flag3d_path, 'subset_video', '*.mp4'))

    for vid in tqdm(videos):

        video_name = vid.split('/')[-1]
        image_path = vid.replace(video_name, video_name.split('.')[0])
        image_path = image_path.replace('video', 'image')

        fps = 30

        os.makedirs(image_path, exist_ok=True)
        os.system(f'ffmpeg -i {vid}  -v quiet -f image2 -r {fps} '
                  f'-b:v 5626k {image_path}{os.path.sep}%04d.png')

        # pdb.set_trace()
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--flag3d_path', type=str, default='/mnt/d/datasets/flag3d')

    args = parser.parse_args()

    reformat_flag3d(args)
    # split_flag3d(args)
    












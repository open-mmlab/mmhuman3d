# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from queue import Queue
from threading import Event, Lock, Thread

import torch
import cv2
import numpy as np
import mmcv
# from mmpose.apis import init_pose_model
from pytorch3d.renderer import FoVOrthographicCameras, FoVPerspectiveCameras
from mmhuman3d.models.builder import build_body_model
from mmpose.utils import StopWatch
from mmpose.core.visualization import imshow_keypoints,imshow_mesh_3d
from mmhuman3d.core.visualization.renderer.torch3d_renderer.utils import tensor2array
from mmhuman3d.core.visualization.renderer import build_renderer
from mmhuman3d.core.conventions.cameras.convert_convention import \
    convert_camera_matrix  # prevent yapf isort conflict
from mmhuman3d.utils.demo_utils import convert_bbox_to_intrinsic, conver_verts_to_cam_coord, get_default_hmr_intrinsic, smooth_process
from mmhuman3d.core.cameras.builder import build_cameras
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from mmhuman3d.utils.transforms import rotmat_to_aa
from mmhuman3d.core.visualization import visualize_kp2d
from mmhuman3d.apis import inference_image_based_model,init_model
from mmhuman3d.core.visualization import visualize_smpl_hmr
from mmhuman3d.utils.demo_utils import process_mmdet_results

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--mesh_reg_config',
    type=str,
    default='configs/spin/resnet50_spin_pw3d.py',
    help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='data/checkpoints/resnet50_spin_pw3d-e1857270_20211201.pth',
        help='Checkpoint file for mesh regression')
    parser.add_argument('--cam-id', type=str, default='0')
    parser.add_argument(
        '--det-config',
        type=str,
        default='demo/mmdetection_cfg/'
        'ssdlite_mobilenetv2_scratch_600e_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det-checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmdetection/v2.0/ssd/'
        'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
        'scratch_600e_coco_20210629_110627-974d9307.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.6,
        help='Bounding box score threshold')
    parser.add_argument(
        '--smooth_type',
        type=str,
        default=None,
        help='Smooth the data through the specified type.'
        'Select in [oneeuro,gaus1d,savgol].')        
    parser.add_argument(
        '--out-video-fps',
        type=int,
        default=20,
        help='Set the FPS of the output video file.')

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=-1,
        help='Frame buffer size. If set -1, the buffer size will be '
        'automatically inferred from the display delay time. Default: -1')

    parser.add_argument(
        '--inference-fps',
        type=int,
        default=10,
        help='Maximum inference FPS. This is to limit the resource consuming '
        'especially when the detection and pose model are lightweight and '
        'very fast. Default: 10.')

    parser.add_argument(
        '--display-delay',
        type=int,
        default=100,
        help='Delay the output video in milliseconds. This can be used to '
        'align the output video and inference results. The delay can be '
        'disabled by setting a non-positive delay time. Default: 0')

    parser.add_argument(
        '--synchronous-mode',
        action='store_true',
        help='Enable synchronous mode that video I/O and inference will be '
        'temporally aligned. Note that this will reduce the display FPS.')

    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
        'See also --smooth-filter-cfg.')

    return parser.parse_args()


def read_camera():
    # init video reader
    print('Thread "input" started')
    cam_id = args.cam_id
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()

    while not event_exit.is_set():
        # capture a camera frame
        ret_val, frame = vid_cap.read()
        if ret_val:
            ts_input = time.time()

            event_inference_done.clear()
            with input_queue_mutex:
                input_queue.append((ts_input, frame))

            if args.synchronous_mode:
                event_inference_done.wait()

            frame_buffer.put((ts_input, frame))
        else:
            # input ending signal
            frame_buffer.put((None, None))
            break

    vid_cap.release()


def inference_detection():
    print('Thread "det" started')
    stop_watch = StopWatch(window=10)
    min_interval = 1.0 / args.inference_fps
    _ts_last = None  # timestamp when last inference was done

    while True:
        while len(input_queue) < 1:
            time.sleep(0.001)
        with input_queue_mutex:
            ts_input, frame = input_queue.popleft()
        # inference detection
        with stop_watch.timeit('Det'):
            mmdet_results = inference_detector(det_model, frame)

        t_info = stop_watch.report_strings()
        with det_result_queue_mutex:
            det_result_queue.append((ts_input, frame, t_info, mmdet_results))

        # limit the inference FPS
        _ts = time.time()
        if _ts_last is not None and _ts - _ts_last < min_interval:
            time.sleep(min_interval - _ts + _ts_last)
        _ts_last = time.time()


def inference_mesh():
    print('Thread "mesh" started')
    stop_watch = StopWatch(window=10)

    while True:
        while len(det_result_queue) < 1:
            time.sleep(0.001)
        with det_result_queue_mutex:
            ts_input, frame, t_info, mmdet_results = det_result_queue.popleft()

        with stop_watch.timeit('Mesh'):
            det_results = process_mmdet_results(
                mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)

            mesh_results = inference_image_based_model(
                mesh_model,
                frame,
                det_results,
                bbox_thr=args.bbox_thr,
                format='xyxy')
            if mesh_results == []:
                continue
        t_info += stop_watch.report_strings()
        with mesh_result_queue_mutex:
            smpl_betas = mesh_results[0]['smpl_beta']
            smpl_poses = mesh_results[0]['smpl_pose']
            if smpl_poses.shape == (24, 3, 3):
                smpl_poses = rotmat_to_aa(smpl_poses).reshape(-1)
            elif smpl_poses.shape == (24, 3):
                smpl_poses = smpl_poses.reshape(-1)
            else:
                raise (f'Wrong shape of `smpl_pose`: {smpl_poses.shape}')

            pred_cams = mesh_results[0]['camera']
            verts = mesh_results[0]['vertices']
            bboxes_xyxy = mesh_results[0]['bbox']
            kp3d = mesh_results[0]['keypoints_3d']

            if args.smooth_type is not None:
                smpl_poses = smooth_process(
                    smpl_poses.reshape(1, 24, 9), smooth_type=args.smooth_type)
                smpl_poses = smpl_poses.reshape(1, 24, 3, 3)
                verts = smooth_process(verts, smooth_type=args.smooth_type)
                                
            kp3d, _ = conver_verts_to_cam_coord(
                kp3d, pred_cams, bboxes_xyxy, focal_length=5000.)
            # K1 = convert_bbox_to_intrinsic(bboxes_xyxy[None])
            # kp3d_224 = np.dot(K0,kp3d.T).squeeze()
            # kp3d_orig = np.dot(K1,kp3d_224).T.squeeze()

            kp2d = np.zeros([17,2])
            kp3d =kp3d.squeeze()

            kp2d[:, 0] = (kp3d[:,0] * 5000+(112*kp3d[:,2]))/kp3d[:,2]
            kp2d[:, 1] = (kp3d[:,1] * 5000+(112*kp3d[:,2]))/kp3d[:,2]

            mesh_result_queue.append((ts_input, t_info, smpl_poses, smpl_betas,
                                      pred_cams, verts, kp2d, bboxes_xyxy))

        event_inference_done.set()


def display(renderer, cameras, textures, faces):
    print('Thread "display" started')
    stop_watch = StopWatch(window=10)

    # initialize result status
    ts_inference = None  # timestamp of the latest inference result
    fps_inference = 0.  # infenrece FPS
    t_delay_inference = 0.  # inference result time delay
    verts = None  # latest inference result
    t_info = []  # upstream time information (list[str])

    # initialize visualization and output
    text_color = (228, 183, 61)  # text color to show time/system information
    vid_out = None  # video writer

    # show instructions
    print('Keyboard shortcuts: ')
    print('"v": Toggle the visualization of bounding boxes and meshes.')
    print('"Q", "q" or Esc: Exit.')

    while True:
        with stop_watch.timeit('_FPS_'):
            # acquire a frame from buffer
            ts_input, frame = frame_buffer.get()
            # input ending signal
            if ts_input is None:
                break

            img = frame

            # get mesh estimation results
            if len(mesh_result_queue) > 0:
                with mesh_result_queue_mutex:
                    _result = mesh_result_queue.popleft()
                    _ts_input, t_info, smpl_poses, smpl_betas, \
                        pred_cams, verts, kp2d, bboxes_xyxy = _result

                _ts = time.time()
                if ts_inference is not None:
                    fps_inference = 1.0 / (_ts - ts_inference)
                ts_inference = _ts
                t_delay_inference = (_ts - _ts_input) * 1000
            if verts is not None:
                # show bounding boxes
                mmcv.imshow_bboxes(
                    img, bboxes_xyxy[None], colors='green', top_k=-1, thickness=2, show=False)

                img = visualize_kp2d(
                    kp2d[None],
                    data_source='h36m',
                    return_array=True,
                    resolution=list(frame.shape[:2]),
                    image_array=frame[None]).squeeze()
                # # visualize smpl
                # if isinstance(verts,np.ndarray):
                #     verts = torch.tensor(verts, dtype=torch.float32)
                # if isinstance(faces,np.ndarray):
                #     faces = torch.tensor(faces)
                # mesh = Meshes(verts, faces[None], textures)
                # rendered_images = renderer(meshes=mesh, cameras=cameras)
                # rgba = renderer.tensor2rgba(rendered_images).cpu().numpy()
                # rgbs, valid_masks = rgba[..., :3] * 255, rgba[..., 3:]
                # img = (rgbs * valid_masks + (1 - valid_masks) * img)[0].astype(np.uint8)
                
            # delay control
            if args.display_delay > 0:
                t_sleep = args.display_delay * 0.001 - (time.time() - ts_input)
                if t_sleep > 0:
                    time.sleep(t_sleep)
            t_delay = (time.time() - ts_input) * 1000

            # show time information
            t_info_display = stop_watch.report_strings()  # display fps
            t_info_display.append(f'Inference FPS: {fps_inference:>5.1f}')
            t_info_display.append(f'Delay: {t_delay:>3.0f}')
            t_info_display.append(
                f'Inference Delay: {t_delay_inference:>3.0f}')
            t_info_str = ' | '.join(t_info_display + t_info)
            cv2.putText(img, t_info_str, (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, text_color, 1)
            # collect system information
            sys_info = [
                f'RES: {img.shape[1]}x{img.shape[0]}',
                f'Buffer: {frame_buffer.qsize()}/{frame_buffer.maxsize}'
            ]
            if psutil_proc is not None:
                sys_info += [
                    f'CPU: {psutil_proc.cpu_percent():.1f}%',
                    f'MEM: {psutil_proc.memory_percent():.1f}%'
                ]
            sys_info_str = ' | '.join(sys_info)
            cv2.putText(img, sys_info_str, (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, text_color, 1)

            # save the output video frame
            # if args.out_video_file is not None:
            #     if vid_out is None:
            #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #         fps = args.out_video_fps
            #         frame_size = (img.shape[1], img.shape[0])
            #         vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
            #                                   frame_size)

            #     vid_out.write(img)

            # display
            cv2.imshow('mmpose webcam demo', img)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input in (27, ord('q'), ord('Q')):
                break
            # elif keyboard_input == ord('v'):
            #     args.vis_mode = (args.vis_mode + 1) % 3

    cv2.destroyAllWindows()
    if vid_out is not None:
        vid_out.release()
    event_exit.set()


def main():
    global args
    global frame_buffer
    global input_queue, input_queue_mutex
    global det_result_queue, det_result_queue_mutex
    global mesh_result_queue, mesh_result_queue_mutex
    global det_model, mesh_model, extractor
    global event_exit, event_inference_done
    global pose_smoother_list
    
    args = parse_args()
    args.synchronous_mode=True
    # args.cam_id = "demo/resources/single_person_demo.mp4"
    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    cam_id = args.cam_id
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()
    _, frame = vid_cap.read()
    resolution = list(frame.shape[:2])
    vid_cap.release()
    # build body model for visualization
    body_model = build_body_model(
                dict(
                    type='SMPL',
                    gender='neutral',
                    num_betas=10,
                    model_path='data/body_models/smpl'))
    faces = torch.tensor(body_model.faces.astype(np.int32)).clone()

    # load visualization config file
    render_cfg = mmcv.Config.fromfile(
        'configs/render/smpl.py')['RENDER_CONFIGS']['lq']
    render_cfg['device'] = args.device
    render_cfg['resolution'] = resolution
    # build renderer
    renderer = build_renderer(render_cfg)
    K = get_default_hmr_intrinsic(num_frame=1, focal_length=5000.)[0]
    K, R, T = convert_camera_matrix(
        convention_dst='pytorch3d',
        K=K,
        R=None,
        T=None,
        is_perspective=True,
        convention_src='opencv',
        resolution_src=resolution,
        in_ndc_src=False,
        in_ndc_dst=False)
    # # camera
    cameras = build_cameras(
        dict(
            type='perspective',
            in_ndc=False,
            device=args.device,
            K=K,
            R=R,
            T=T,
            resolution=resolution)).to(args.device)
    # cameras = FoVPerspectiveCameras(R=torch.eye(3), T=torch.zeros(3)).to(args.device)
    # textures
    textures = TexturesVertex(verts_features=torch.ones([1,6890,3])).to(args.device)


    # build detection model
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())


    # build human3d models

    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # store mesh history for tracking
    # mesh_history_list = [{'mesh_results_last': [], 'next_id': 0}]

    # frame buffer
    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        # infer buffer size from the display delay time
        # assume that the maximum video fps is 30
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    # queue of input frames
    # element: (timestamp, frame)
    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    # queue of detection results
    # element: tuple(timestamp, frame, time_info, det_results)
    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    # queue of detection/pose results
    # element: (timestamp, time_info, pose_results_list)
    mesh_result_queue = deque(maxlen=1)
    mesh_result_queue_mutex = Lock()

    try:
        event_exit = Event()
        event_inference_done = Event()
        t_input = Thread(target=read_camera, args=())
        t_det = Thread(target=inference_detection, args=(), daemon=True)
        t_mesh = Thread(target=inference_mesh, args=(), daemon=True)

        t_input.start()
        t_det.start()
        t_mesh.start()

        # run display in the main thread
        display(renderer, cameras, textures, faces.clone())
        # join the input thread (non-daemon)
        t_input.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

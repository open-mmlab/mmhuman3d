import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint

from mmhuman3d.data.datasets.pipelines import Compose
from mmhuman3d.models import build_architecture, build_backbone
from mmhuman3d.utils.demo_utils import box2cs, xywh2xyxy, xyxy2xywh


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed model.
        (nn.Module, None): The constructed extractor model
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.data.test.test_mode = True

    model = build_architecture(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()

    extractor = None
    if config.model.type == 'VideoBodyModelEstimator':
        extractor = build_backbone(config.extractor.backbone)
        if config.extractor.checkpoint is not None:
            # load model checkpoint
            load_checkpoint(extractor, config.extractor.checkpoint)
        extractor.cfg = config
        extractor.to(device)
        extractor.eval()
    return model, extractor


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='bgr'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the image_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['image_path'], str):
            results['image_file'] = results['image_path']
            img = mmcv.imread(results['image_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['image_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['image_path'], cv2.COLOR_BGR2RGB)
            else:
                img = results['image_path']
        else:
            raise TypeError('"image_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results


def inference_image_based_model(
    model,
    img_or_path,
    det_results,
    bbox_thr=None,
    format='xywh',
):
    """Inference a single image with a list of person bounding boxes.

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (Union[str, np.ndarray]): Image filename or loaded image.
        det_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr (float, optional): Threshold for bounding boxes.
            Only bboxes with higher scores will be fed into the pose detector.
            If bbox_thr is None, ignore it. Defaults to None.
        format (str, optional): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score]),
            SMPL parameters, vertices, kp3d, and camera.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']
    mesh_results = []
    if len(det_results) == 0:
        return []

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    inference_pipeline = [LoadImage()] + cfg.inference_pipeline
    inference_pipeline = Compose(inference_pipeline)

    assert len(bboxes[0]) in [4, 5]

    batch_data = []
    input_size = cfg['img_res']
    aspect_ratio = 1 if isinstance(input_size,
                                   int) else input_size[0] / input_size[1]

    for i, bbox in enumerate(bboxes_xywh):
        center, scale = box2cs(bbox, aspect_ratio, bbox_scale_factor=1.25)
        # prepare data
        data = {
            'image_path': img_or_path,
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'sample_idx': i,
        }
        data = inference_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)

    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        results = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            sample_idx=batch_data['sample_idx'],
        )

    for idx in range(len(det_results)):
        mesh_result = det_results[idx].copy()
        mesh_result['bbox'] = bboxes_xyxy[idx]
        mesh_result['camera'] = results['camera'][idx]
        mesh_result['smpl_pose'] = results['smpl_pose'][idx]
        mesh_result['smpl_beta'] = results['smpl_beta'][idx]
        mesh_result['vertices'] = results['vertices'][idx]
        mesh_result['keypoints_3d'] = results['keypoints_3d'][idx]
        mesh_results.append(mesh_result)
    return mesh_results


def inference_video_based_model(model,
                                extracted_results,
                                with_track_id=True,
                                causal=True):
    """Inference SMPL parameters from extracted featutres using a video-based
    model.

    Args:
        model (nn.Module): The loaded mesh estimation model.
        extracted_results (List[List[Dict]]): Multi-frame feature extraction
            results stored in a nested list. Each element of the outer list
            is the feature extraction results of a single frame, and each
            element of the inner list is the feature information of one person,
            which contains:
                features (ndarray): extracted features
                track_id (int): unique id of each person, required when
                    ``with_track_id==True```
                bbox ((4, ) or (5, )): left, right, top, bottom, [score]
        with_track_id: If True, the element in extracted_results is expected to
            contain "track_id", which will be used to gather the feature
            sequence of a person from multiple frames. Otherwise, the extracted
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
        causal (bool): If True, the target frame is the first frame in
            a sequence. Otherwise, the target frame is in the middle of a
            sequence.

    Returns:
        list[dict]: Each item in the list is a dictionary, which contains:
            SMPL parameters, vertices, kp3d, and camera.
    """
    cfg = model.cfg
    device = next(model.parameters()).device
    seq_len = cfg.data.test.seq_len
    mesh_results = []
    # build the data pipeline
    inference_pipeline = Compose(cfg.inference_pipeline)
    target_idx = 0 if causal else len(extracted_results) // 2

    input_features = _gather_input_features(extracted_results)
    feature_sequences = _collate_feature_sequence(input_features,
                                                  with_track_id, target_idx)
    if not feature_sequences:
        return mesh_results

    batch_data = []

    for i, seq in enumerate(feature_sequences):

        data = {
            'features': seq['features'],
            'sample_idx': i,
        }

        data = inference_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['features'] = batch_data['features'].to(device)

    with torch.no_grad():
        results = model(
            features=batch_data['features'],
            img_metas=batch_data['img_metas'],
            sample_idx=batch_data['sample_idx'])

    results['camera'] = results['camera'].reshape(-1, seq_len, 3)
    results['smpl_pose'] = results['smpl_pose'].reshape(-1, seq_len, 24, 3, 3)
    results['smpl_beta'] = results['smpl_beta'].reshape(-1, seq_len, 10)
    results['vertices'] = results['vertices'].reshape(-1, seq_len, 6890, 3)
    results['keypoints_3d'] = results['keypoints_3d'].reshape(
        -1, seq_len, 17, 3)

    for idx in range(len(feature_sequences)):
        mesh_result = dict()
        # mesh_result['track_id'] = feature_sequences[idx]['track_id']
        mesh_result['camera'] = results['camera'][idx, target_idx]
        mesh_result['smpl_pose'] = results['smpl_pose'][idx, target_idx]
        mesh_result['smpl_beta'] = results['smpl_beta'][idx, target_idx]
        mesh_result['vertices'] = results['vertices'][idx, target_idx]
        mesh_result['keypoints_3d'] = results['keypoints_3d'][idx, target_idx]
        mesh_results.append(mesh_result)
    return mesh_results


def feature_extract(
    model,
    img_or_path,
    det_results,
    bbox_thr=None,
    format='xywh',
):
    """Extract image features with a list of person bounding boxes.

    Args:
        model (nn.Module): The loaded feature extraction model.
        img_or_path (Union[str, np.ndarray]): Image filename or loaded image.
        det_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr (float, optional): Threshold for bounding boxes.
            If bbox_thr is None, ignore it. Defaults to None.
        format (str, optional): bbox format. Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: The bbox & pose info,
            containing the bbox: (left, top, right, bottom, [score])
            and the features.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    cfg = model.cfg
    device = next(model.parameters()).device

    feature_results = []
    if len(det_results) == 0:
        return feature_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in det_results])
    assert len(bboxes[0]) in [4, 5]

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        det_results = [det_results[i] for i in valid_idx]

    # if bbox_thr remove all bounding box
    if len(bboxes) == 0:
        return feature_results

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = xywh2xyxy(bboxes)

    # build the data pipeline
    extractor_pipeline = [LoadImage()] + cfg.extractor_pipeline
    extractor_pipeline = Compose(extractor_pipeline)
    batch_data = []
    input_size = cfg['img_res']
    aspect_ratio = 1 if isinstance(input_size,
                                   int) else input_size[0] / input_size[1]

    for i, bbox in enumerate(bboxes_xywh):
        center, scale = box2cs(bbox, aspect_ratio, bbox_scale_factor=1.25)
        # prepare data
        data = {
            'image_path': img_or_path,
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'sample_idx': i,
        }
        data = extractor_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)

    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        results = model(batch_data['img'])

        if isinstance(results, list) or isinstance(results, tuple):
            results = results[-1].mean(dim=-1).mean(dim=-1)

    for idx in range(len(det_results)):
        feature_result = det_results[idx].copy()
        feature_result['bbox'] = bboxes_xyxy[idx]
        feature_result['features'] = results[idx].cpu().numpy()
        feature_results.append(feature_result)

    return feature_results


def _gather_input_features(extracted_results):
    """Gather input features.

    Args:
        extracted_results (List[List[Dict]]):
            Multi-frame feature extraction results

    Returns:
        List[List[dict]]: Multi-frame feature extraction results
            stored in a nested list. Each element of the outer list is the
            feature extraction results of a single frame, and each element of
            the inner list is the extracted results of one person,
            which contains:
                features (ndarray): extracted features
                track_id (int): unique id of each person, required when
                    ``with_track_id==True```
    """
    sequence_inputs = []
    for frame in extracted_results:
        frame_inputs = []
        for res in frame:
            inputs = dict()
            if 'features' in res:
                inputs['features'] = res['features']
            if 'track_id' in res:
                inputs['track_id'] = res['track_id']
            frame_inputs.append(inputs)
        sequence_inputs.append(frame_inputs)
    return sequence_inputs


def _collate_feature_sequence(extracted_features,
                              with_track_id=True,
                              target_frame=0):
    """Reorganize multi-frame feature extraction results into individual
    feature sequences.

    Args:
        extracted_features (List[List[Dict]]): Multi-frame feature extraction
            results stored in a nested list. Each element of the outer list
            is the feature extraction results of a single frame, and each
            element of the inner list is the extracted results of one person,
            which contains:
                features (ndarray): extracted features
                track_id (int): unique id of each person, required when
                    ``with_track_id==True```
        with_track_id (bool): If True, the element in pose_results is expected
            to contain "track_id", which will be used to gather the pose
            sequence of a person from multiple frames. Otherwise, the pose
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
        target_frame (int): The index of the target frame. Default: 0.
    """
    T = len(extracted_features)
    assert T > 0

    target_frame = (T + target_frame) % T  # convert negative index to positive

    N = len(
        extracted_features[target_frame])  # use identities in the target frame
    if N == 0:
        return []

    C = extracted_features[target_frame][0]['features'].shape[0]

    track_ids = None
    if with_track_id:
        track_ids = [
            res['track_id'] for res in extracted_features[target_frame]
        ]

    feature_sequences = []
    for idx in range(N):
        feature_seq = dict()
        # gather static information
        for k, v in extracted_features[target_frame][idx].items():
            if k != 'features':
                feature_seq[k] = v
        # gather keypoints
        if not with_track_id:
            feature_seq['features'] = np.stack(
                [frame[idx]['features'] for frame in extracted_features])
        else:
            features = np.zeros((T, C), dtype=np.float32)
            features[target_frame] = extracted_features[target_frame][idx][
                'features']
            # find the left most frame containing track_ids[idx]
            for frame_idx in range(target_frame - 1, -1, -1):
                contains_idx = False
                for res in extracted_features[frame_idx]:
                    if res['track_id'] == track_ids[idx]:
                        features[frame_idx] = res['features']
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the left most frame
                    features[frame_idx] = features[frame_idx + 1]

            # find the right most frame containing track_idx[idx]
            for frame_idx in range(target_frame + 1, T):
                contains_idx = False
                for res in extracted_features[frame_idx]:
                    if res['track_id'] == track_ids[idx]:
                        features[frame_idx] = res['features']
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the right most frame
                    features[frame_idx] = features[frame_idx - 1]
                    # break
            feature_seq['features'] = features
        feature_sequences.append(feature_seq)

    return feature_sequences

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint

from mmhuman3d.data.datasets.pipelines import Compose
from mmhuman3d.models import build_architecture
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
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.data.test.test_mode = True
    # config.model.pretrained = None
    model = build_architecture(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


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


def inference_model(
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
        list[dict]: The bbox & pose info,
            Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
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
        x, y, w, h, _ = bbox
        center, scale = box2cs(x, y, w, h, aspect_ratio)
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
        mesh_result['center'] = batch_data['img_metas'][idx]['center']
        mesh_result['scale'] = batch_data['img_metas'][idx]['scale']
        mesh_result['camera'] = results['camera'][idx]
        mesh_result['smpl_pose'] = results['smpl_pose'][idx]
        mesh_result['smpl_beta'] = results['smpl_beta'][idx]
        mesh_result['vertices'] = results['vertices'][idx]
        mesh_result['keypoints_3d'] = results['keypoints_3d'][idx]
        mesh_results.append(mesh_result)
    return mesh_results

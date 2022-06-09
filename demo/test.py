from mmhuman3d.data.data_structures.human_data import HumanData
import requests
import json
import numpy as np
if __name__ == "__main__":
    result_path = '/home/SENSETIME/fanxiangyu/code/mmhuman3d/data/demo_result/inference_result.npz'
    data = HumanData.fromfile(result_path)
    for k,v in data['smpl'].items():
        data['smpl'][k] = v.tolist()
    data['smpl']['transl'] = np.zeros([36,3]).tolist()
    params = {
        "human_data":{'smpl':data['smpl']},
        "actor_type":"MXJBald",
        # "actor_name":"rp_yasmin_rigged_009",
        "motion_category":"human_data",
        "regenerate":False,
        "output_smplx":False,
        "output_bbox":False,
        "output_fbx":True,
        "output_mo_json":True,

    }
    retarget_url = 'http://10.10.30.159:8763/api/v1/retargeting/retarget/'
    resp = requests.post(retarget_url, json=params)
    resp_json = resp.json()
    if not resp_json.get('code') == 200:
        err_msg = (
            f'Error response from POST {retarget_url}'
            f'Response:\n{resp_json}'
        )
        # print(err_msg)
        raise RuntimeError(err_msg)
    resp_content = resp_json['content']

    # assert 'motion_url' in resp_content, f'{resp_content}'
    # assert 'bbox_url' in resp_content, f'{resp_content}'
    # resp_content['mo_json'] 

    print(resp_content['motion_url'])
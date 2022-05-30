from mmhuman3d.data.data_structures.human_data import HumanData
import requests
import json

if __name__ == "__main__":
    result_path = 'data/demo_result/inference_result.npz'
    data = HumanData.fromfile(result_path)
    for k,v in data['smpl'].items():
        data['smpl'][k] = v.tolist()
    params = {
        "human_data":data['smpl'],
        "actor_type":"Mannequin",
        "actor_name":"rp_yasmin_rigged_009",
        "motion_category":"humandata",
        "regenerate":True,
        "output_smpl":False
    }
    retarget_url = 'http://10.152.237.9:7000/retarget/'
    resp = requests.post(retarget_url, json=json.dumps(params))
    if not resp_json.get('code') == 200:
        err_msg = (
            f'Error response from POST {self.retarget_url} with params: {params}.\n'
            f'Response:\n{resp_json}'
        )
        print(err_msg)
        raise RuntimeError(err_msg)
    resp_content = resp_json['content']

    assert 'motion_url' in resp_content, f'{resp_content}'
    assert 'bbox_url' in resp_content, f'{resp_content}'
    fbx_url, bbox_url = resp_content['motion_url'], resp_content['bbox_url']

    print(fbx_url,bbox_url)
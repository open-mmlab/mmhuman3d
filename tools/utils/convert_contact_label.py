
import numpy as np
import pickle
import json

import pdb

vertex2part_smplx_dict = dict(
    left_foot = [5791, 5794, 5795, 5807, 5814, 5827, 5828, 5839, 5842, 5848, 5850, 
                5851, 5852, 5854, 5855, 5861, 5862, 5865, 5906, 5907, 5908, 5909, 
                5912, 5913, 5914, 5915, 5916, 5917, 8858, 8863, 8864, 8866, 8867, 
                8868, 8881, 8882, 8885, 8886, 8887, 8890, 8897, 8898, 8899, 8900, 
                8901, 8902, 8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 
                8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 
                8923, 8924, 8925, 8929, 8930, 8934],
    right_foot = [8485, 8488, 8489, 8497, 8500, 8501, 8507, 8508, 8544, 8545, 8546, 
             8548, 8549, 8555, 8559, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 
             8607, 8608, 8609, 8610, 8611, 8651, 8652, 8654, 8655, 8656, 8669, 
             8675, 8676, 8677, 8678, 8679, 8685, 8686, 8687, 8688, 8689, 8690, 
             8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 
             8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712, 
             8713, 8714, 8715, 8716])

vertex2part_smpl_dict = dict(
    left_foot = [3237, 3239, 3241, 3253, 3260, 3274, 3275, 3287, 3288, 3294, 3296, 
                 3297, 3298, 3300, 3301, 3307, 3308, 3310, 3352, 3353, 3354, 3355, 
                 3358, 3359, 3360, 3361, 3362, 3363, 3398, 3403, 3405, 3406, 3407, 
                 3408, 3421, 3422, 3425, 3426, 3427, 3430, 3437, 3438, 3439, 3440, 
                 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 
                 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 
                 3463, 3464, 3465, 3466, 3467, 3468],
    right_foot = [6637, 6640, 6641, 6649, 6652, 6653, 6661, 6662, 6696, 6697, 6698, 
                  6700, 6701, 6707, 6711, 6752, 6753, 6754, 6755, 6756, 6757, 6758, 
                  6759, 6760, 6761, 6762, 6763, 6803, 6804, 6806, 6807, 6808, 6821, 
                  6827, 6828, 6829, 6830, 6831, 6837, 6838, 6839, 6840, 6841, 6842, 
                  6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853, 
                  6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 
                  6865, 6866, 6867, 6868])

# load smplx_vert2region
smplx_vert2region = np.load('tools/utils/smplx_vert2region.npy')
smpl_vert2region = np.load('tools/utils/smpl_vert2region.npy')


def get_contact_label_from_smplx_vertex(smplx_vertex_annot, 
                                        smplx_vertex2part_smplx_dict=vertex2part_smplx_dict):
    """
    smplx_vertex: np.array of shape (10475, 1)
    smplx_vertex2part_smplx_dict: dict of shape {part_name: [vertex_id]}
    """
    contact_dict = {}

    assert smplx_vertex_annot.shape == (10475, 1)

    for part_name, vertex_ids in smplx_vertex2part_smplx_dict.items():
        part_vertex = np.zeros((smplx_vertex_annot.shape[0], 1))
        part_vertex[vertex_ids] = 1

        if np.sum(smplx_vertex_annot * part_vertex) > 0:
            contact_dict[part_name] = 1
        else:
            contact_dict[part_name] = 0

    return contact_dict

def get_contact_label_from_smpl_vertex(smpl_vertex_annot, 
                                       smpl_vertex2part_smpl_dict=vertex2part_smpl_dict):
    """
    smpl_vertex: np.array of shape (6890, 1)
    smpl_vertex2part_smpl_dict: dict of shape {part_name: [vertex_id]}
    """
    contact_dict = {}

    assert smpl_vertex_annot.shape == (6890, 1)

    for part_name, vertex_ids in smpl_vertex2part_smpl_dict.items():
        part_vertex = np.zeros((smpl_vertex_annot.shape[0], 1))
        part_vertex[vertex_ids] = 1

        if np.sum(smpl_vertex_annot * part_vertex) > 0:
            contact_dict[part_name] = 1
        else:
            contact_dict[part_name] = 0

    return contact_dict

def get_contact_region_label_from_smplx_vertex(smplx_vertex_annot,
                                                smplx_vertex2region=smplx_vert2region):
        """
        smplx_vertex: np.array of shape (10475, 1)
        smplx_vertex2region: np.array of shape (10475, 75)
        """
        contact_region = np.zeros((75))
    
        assert smplx_vertex_annot.shape == (10475, 1)
    
        for region_id in range(smplx_vertex2region.shape[1]):
            region_vertex = smplx_vertex2region[:, region_id].reshape(-1, 1)
    
            if np.sum(smplx_vertex_annot * region_vertex) > 0:
                contact_region[region_id] = 1
            else:
                contact_region[region_id] = 0
    
        return contact_region

def get_contact_region_label_from_smpl_vertex(smpl_vertex_annot,
                                                smpl_vertex2region=smpl_vert2region):
        """
        smpl_vertex: np.array of shape (6890, 1)
        smpl_vertex2region: np.array of shape (6890, 75)
        """
        contact_region = np.zeros((75))
    
        assert smpl_vertex_annot.shape == (6890, 1)
    
        for region_id in range(smpl_vertex2region.shape[1]):
            region_vertex = smpl_vertex2region[:, region_id].reshape(-1, 1)
    
            if np.sum(smpl_vertex_annot * region_vertex) > 0:
                contact_region[region_id] = 1
            else:
                contact_region[region_id] = 0
    
        return contact_region

if __name__ == '__main__':

    smplx2smpl = pickle.load(open('tools/utils/smplx2smpl.pkl', 'rb'))
    # smplx_model = dict(np.load(f'data/body_models/smplx/SMPLX_NEUTRAL.npz', allow_pickle=True))
    # import smplx
    # smplx_model = smplx.create(
    #     'data/body_models/smplx/SMPLX_NEUTRAL.npz',
    #     model_type='smplx',
    # )
    # output = smplx_model(return_verts=True)
    # faces = smplx_model.faces

    # CR2smplx_f = json.load(open('tools/utils/contact_regions.json', 'r'))['rid_to_smplx_fids']

    # smplx_vert2region = np.zeros((10475, 75))

    # # create vertex2face dict
    # for region_id in range(len(CR2smplx_f)):
    #     region_fids = CR2smplx_f[region_id]

    #     vert_ids = []
    #     for fid in region_fids:
    #         vert_ids += faces[fid].tolist()
    #     vert_ids = list(set(vert_ids))

    #     smplx_vert2region[vert_ids, region_id] = 1

    # save to npy
    # np.save('tools/utils/smplx_vert2region.npy', smplx_vert2region)

    # import smplx
    # smplx_model = smplx.create(
    #     'data/body_models/smplx/SMPLX_NEUTRAL.npz',
    #     model_type='smplx',
    # )
    # output = smplx_model(return_verts=True)
    # faces = smplx_model.faces

    # CR2smplx_f = json.load(open('tools/utils/contact_regions.json', 'r'))['rid_to_smplx_fids']

    # smpl_vert2region = np.zeros((6890, 75))

    # # create vertex2face dict
    # for region_id in range(len(CR2smplx_f)):
    #     region_fids = CR2smplx_f[region_id]

    #     vert_ids = []
    #     for fid in region_fids:
    #         vert_ids += faces[fid].tolist()

    #     vert_ids = list(set(vert_ids))
    #     smplx_verts = np.zeros((10475, 1))
    #     smplx_verts[vert_ids] = 1
    #     # downsample to smpl
    #     smpl_vertex_all = (smplx2smpl['matrix'] @ smplx_verts) > 0.5
    #     smpl_vertex_all = smpl_vertex_all.reshape(-1)
    #     # pdb.set_trace()
    #     smpl_vert2region[smpl_vertex_all, region_id] = 1

    # # save to npy
    # np.save('tools/utils/smpl_vert2region.npy', smpl_vert2region)

    pdb.set_trace()

    # for part_name, smplx_vertex_ids in vertex2part_smplx_dict.items():
    #     part_vertex = np.zeros((10475, 1))
    #     part_vertex[smplx_vertex_ids] = 1
    #     # downsample to smpl
    #     smpl_vertex_all = smplx2smpl['matrix'] @ part_vertex
        
    #     smpl_vertex_ids = [s for s, ann in enumerate(smpl_vertex_all) if ann > 0.5]

    #     print(part_name, smpl_vertex_ids)

    #     pdb.set_trace()



import torch

from mmhuman3d.models.body_models.builder import build_body_model

body_model_load_dir = 'data/body_models/smpl'
extra_joints_regressor = 'data/body_models/J_regressor_extra.npy'


def test_smpl():

    random_body_pose = torch.rand((1, 69))

    # test SMPL
    smpl_54 = build_body_model(
        dict(
            type='SMPL',
            keypoint_src='smpl_54',
            keypoint_dst='smpl_54',
            model_path=body_model_load_dir,
            extra_joints_regressor=extra_joints_regressor))

    smpl_54_output = smpl_54(body_pose=random_body_pose)
    smpl_54_joints = smpl_54_output['joints']

    smpl_49 = build_body_model(
        dict(
            type='SMPL',
            keypoint_src='smpl_54',
            keypoint_dst='smpl_49',
            keypoint_approximate=True,
            model_path=body_model_load_dir,
            extra_joints_regressor=extra_joints_regressor))

    smpl_49_output = smpl_49(body_pose=random_body_pose)
    smpl_49_joints = smpl_49_output['joints']

    joint_mapping = [
        24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47,
        48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27
    ]

    assert torch.allclose(smpl_54_joints[:, joint_mapping, :], smpl_49_joints)


# def test_gendered_smpl():
#     random_betas_neutral = torch.rand((1, 10))
#     random_betas_male = torch.rand((1, 10))
#     random_betas_female = torch.rand((1, 10))
#     gender = torch.Tensor([-1, 0, 1])
#
#     smpl_neutral = build_body_model(
#         dict(
#             type='SMPL',
#             gender='neutral',
#             keypoint_src='smpl_45',
#             keypoint_dst='smpl_45',
#             model_path=body_model_load_dir,
#         ))
#
#     smpl_male = build_body_model(
#         dict(
#             type='SMPL',
#             gender='male',
#             keypoint_src='smpl_45',
#             keypoint_dst='smpl_45',
#             model_path=body_model_load_dir,
#         ))
#
#     smpl_female = build_body_model(
#         dict(
#             type='SMPL',
#             gender='female',
#             keypoint_src='smpl_45',
#             keypoint_dst='smpl_45',
#             model_path=body_model_load_dir,
#         ))
#
#     gendered_smpl = build_body_model(
#         dict(
#             type='GenderedSMPL',
#             keypoint_src='smpl_45',
#             keypoint_dst='smpl_45',
#             model_path=body_model_load_dir))
#
#     smpl_neutral_output = smpl_neutral(betas=random_betas_neutral)
#     smpl_male_output = smpl_male(betas=random_betas_male)
#     smpl_female_output = smpl_female(betas=random_betas_female)
#
#     betas_concat = torch.cat(
#         [random_betas_neutral, random_betas_male, random_betas_female])
#     joint_concat = torch.cat([
#         smpl_neutral_output['joints'], smpl_male_output['joints'],
#         smpl_female_output['joints']
#     ])
#
#     gendered_smpl_output = gendered_smpl(betas=betas_concat, gender=gender)
#
#     assert torch.allclose(joint_concat, gendered_smpl_output['joints'])

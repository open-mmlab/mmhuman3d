from mmhuman3d.models.body_models.star import STAR

body_model_load_dir = 'data/body_models/star'


def test_star_init():
    _ = STAR(model_path=body_model_load_dir)

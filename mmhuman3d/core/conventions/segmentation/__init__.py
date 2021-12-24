from .smpl import SMPL_SEGMENTATION_DICT, SMPL_SUPER_SET
from .smplx import SMPLX_SEGMENTATION_DICT, SMPLX_SUPER_SET


class body_segmentation(object):
    """SMPL(X) body mesh vertex segmentation."""

    def __init__(self, model_type='smpl') -> None:
        if model_type == 'smpl':
            self.DICT = SMPL_SEGMENTATION_DICT
            self.super_set = SMPL_SUPER_SET
            self.NUM_VERTS = 6890
        elif model_type == 'smplx':
            self.DICT = SMPLX_SEGMENTATION_DICT
            self.super_set = SMPLX_SUPER_SET
            self.NUM_VERTS = 10475
        else:
            raise ValueError(f'Wrong model_type: {model_type}.'
                             f' Should be in {["smpl", "smplx"]}')
        self.model_type = model_type
        self.len = len(list(self.DICT))

    def items(self, ):
        return zip(self.keys(), [self.__getitem__(key) for key in self.keys()])

    def keys(self, ):
        return self.DICT.keys()

    def values(self, ):
        return [self.__getitem__(key) for key in self.keys()]

    def __len__(self, ):
        return self.len

    def __getitem__(self, key):
        if key in self.DICT.keys():
            part_segmentation = []
            raw_segmentation = self.DICT[key]
            for continuous in raw_segmentation:
                if len(continuous) == 2:
                    part_segmentation.extend(
                        list(range(continuous[0], continuous[1] + 1)))
                elif len(continuous) == 1:
                    part_segmentation.extend(continuous)
            return part_segmentation
        elif key in self.super_set.keys():
            super_part_segmentation = []
            for body_part_key in self.super_set[key]:
                super_part_segmentation += self.__getitem__(body_part_key)
            return super_part_segmentation
        elif key.lower() == 'all':
            return list(range(self.NUM_VERTS))
        else:
            raise KeyError(f'{key} not in {self.model_type} conventions.')


def _preprocess_segmentation_dict(segmentation_dict):
    """help to preprocess the indexes to list."""
    final_dict = {}
    for k in segmentation_dict:
        final_dict[k] = [[]]
        final_part_indexes = final_dict[k]
        part_indexes = segmentation_dict[k]
        part_indexes.sort()
        for index in range(len(part_indexes)):
            if len(final_part_indexes[-1]) == 0:
                final_part_indexes[-1].append(part_indexes[index])
            elif len(final_part_indexes[-1]) == 2:
                final_part_indexes.append([part_indexes[index]])
            elif len(final_part_indexes[-1]) == 1:
                if index != len(part_indexes) - 1:
                    this_index = part_indexes[index]
                    last_index = part_indexes[index - 1]
                    next_index = part_indexes[index + 1]
                    if (this_index == last_index + 1) and (this_index
                                                           == next_index - 1):
                        pass
                    elif (this_index == last_index +
                          1) and (this_index != next_index - 1):
                        final_part_indexes[-1].append(this_index)
                    elif (this_index != last_index + 1) and (this_index !=
                                                             next_index - 1):
                        final_part_indexes.append([this_index])
                        final_part_indexes.append([])
                    elif (this_index !=
                          last_index + 1) and (this_index == next_index - 1):
                        final_part_indexes.append([this_index])
                else:
                    this_index = part_indexes[index]
                    last_index = part_indexes[index - 1]
                    if (this_index == last_index + 1):
                        final_part_indexes[-1].append(this_index)
                    else:
                        final_part_indexes.append([this_index])
    return final_dict

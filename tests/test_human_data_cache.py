from mmhuman3d.data.data_structures.human_data import HumanData

# from mmhuman3d.data.data_structures.human_data_cache import (
#     HumanDataCacheReader,
#     HumanDataCacheWriter,
# )

human_data_load_path = '/Users/gaoyang/Downloads/h36m_train.npz'
human_data_cache_path = '/Users/gaoyang/Downloads/h36m_cache.npz'


def main():
    human_data = HumanData.fromfile(human_data_load_path)
    print(human_data.temporal_len)
    # for slice_size in (1, 10, 100):
    #     writer_kwargs, sliced_data = human_data.get_sliced_cache(
    #          slice_size=slice_size)
    #     writer = HumanDataCacheWriter(**writer_kwargs)
    #     writer.update_sliced_dict(sliced_data)
    #     writer.dump(human_data_cache_path.replace(
    #       '.npz', f'_{slice_size}.npz'))


main()

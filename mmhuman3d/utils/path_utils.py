from pathlib import Path


def check_path_suffix(path_str, allowed_suffix=[]):
    """Check whether the suffix of the path is allowed.

    Args:
        path_str (str):
            Path to check.
        allowed_suffix (list, optional):
            What extension names are allowed.
            Offer a list like ['.jpg', ',jpeg'].
            When it's [], only a blank suffix is allowed.
            Defaults to [].

    Returns:
        int:
            0: suffix test passed
            1: suffix test failed
    """
    pathinfo = Path(path_str)
    suffix = pathinfo.suffix.lower()
    if suffix == '':
        if len(allowed_suffix) == 0:
            return 0
        else:
            return 1
    else:
        for index, tmp_suffix in enumerate(allowed_suffix):
            if not tmp_suffix.startswith('.'):
                tmp_suffix = '.' + tmp_suffix
            allowed_suffix[index] = tmp_suffix.lower()
        if suffix in allowed_suffix:
            return 0
        else:
            return 1


def check_path_existence(
    path_str,
    path_type='auto',
):
    """Check whether a file or a directory exist at the expected path.

    Args:
        path_str (str):
            Path to check.
        path_type (str, optional):
            What kind of file do we expect at the path.
            Choose among file, directory, auto.
            Defaults to 'auto'.

    Raises:
        KeyError: if path_type conflicts with path_str

    Returns:
        int:
            0: File at path_str matches path_type, and it exists.
            1: Its parent doesn't exist.
            2: Expecting a folder at path_str, but not found.
            3: Expecting a file at path_str, but not found.
    """
    path_type = path_type.lower()
    assert path_type in ['file', 'directory', 'auto']
    pathinfo = Path(path_str)
    if not pathinfo.parent.is_dir():
        return 1
    suffix = pathinfo.suffix.lower()
    if path_type == 'directory' or\
            path_type == 'auto' and suffix == '':
        if pathinfo.exists() and pathinfo.is_dir():
            return 0
        else:
            return 2
    elif path_type == 'file' or\
            path_type == 'auto' and suffix != '':
        if not pathinfo.exists() or not pathinfo.is_file():
            return 3
        else:
            return 0
    else:
        raise KeyError(f'{path_str} doesn\'t match any expectation '
                       f'when type is set to: {path_type}')

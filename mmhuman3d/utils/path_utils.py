from enum import Enum
from pathlib import Path
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def check_path_suffix(path_str: str, allowed_suffix: List[str] = []) -> bool:
    """Check whether the suffix of the path is allowed.

    Args:
        path_str (str):
            Path to check.
        allowed_suffix (List[str], optional):
            What extension names are allowed.
            Offer a list like ['.jpg', ',jpeg'].
            When it's [], only a blank suffix is allowed.
            Defaults to [].

    Returns:
        bool:
            True: suffix test passed
            False: suffix test failed
    """
    pathinfo = Path(path_str)
    suffix = pathinfo.suffix.lower()
    if suffix == '':
        if len(allowed_suffix) == 0:
            return True
        else:
            return False
    else:
        for index, tmp_suffix in enumerate(allowed_suffix):
            if not tmp_suffix.startswith('.'):
                tmp_suffix = '.' + tmp_suffix
            allowed_suffix[index] = tmp_suffix.lower()
        if suffix in allowed_suffix:
            return True
        else:
            return False


class Existence(Enum):
    Exist = 0
    MissingParent = 1
    FolderNotExist = 2
    FileNotExist = 3


def check_path_existence(
        path_str: str,
        path_type: Literal['file', 'directory', 'auto'] = 'auto') -> Existence:
    """Check whether a file or a directory exist at the expected path.

    Args:
        path_str (str):
            Path to check.
        path_type (Literal[, optional):
            What kind of file do we expect at the path.
            Choose among file, directory, auto.
            Defaults to 'auto'.

    Raises:
        KeyError: if path_type conflicts with path_str

    Returns:
        Existence:
            0: Exist, file at path_str matches path_type, and it exists.
            1: MissingParent, its parent doesn't exist.
            2: FolderNotExist, expecting a folder at path_str, but not found.
            3: FileNotExist, expecting a file at path_str, but not found.
    """
    path_type = path_type.lower()
    assert path_type in ['file', 'directory', 'auto']
    pathinfo = Path(path_str)
    if not pathinfo.parent.is_dir():
        return Existence.MissingParent
    suffix = pathinfo.suffix.lower()
    if path_type == 'directory' or\
            path_type == 'auto' and suffix == '':
        if pathinfo.exists() and pathinfo.is_dir():
            return Existence.Exist
        else:
            return Existence.FolderNotExist
    elif path_type == 'file' or\
            path_type == 'auto' and suffix != '':
        if not pathinfo.exists() or not pathinfo.is_file():
            return Existence.FileNotExist
        else:
            return Existence.Exist
    else:
        raise KeyError(f'{path_str} doesn\'t match any expectation '
                       f'when type is set to: {path_type}')

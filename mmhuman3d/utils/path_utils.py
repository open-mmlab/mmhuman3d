import os
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def check_path_suffix(path_str: str,
                      allowed_suffix: Union[str, List[str]] = '') -> bool:
    """Check whether the suffix of the path is allowed.

    Args:
        path_str (str):
            Path to check.
        allowed_suffix (List[str], optional):
            What extension names are allowed.
            Offer a list like ['.jpg', ',jpeg'].
            When it's [], all will be received.
            Use [''] then directory is allowed.
            Defaults to [].

    Returns:
        bool:
            True: suffix test passed
            False: suffix test failed
    """
    if isinstance(allowed_suffix, str):
        allowed_suffix = [allowed_suffix]
    pathinfo = Path(path_str)
    suffix = pathinfo.suffix.lower()
    if len(allowed_suffix) == 0:
        return True
    if pathinfo.is_dir():
        if '' in allowed_suffix:
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
    """State of file existence."""
    FileExist = 0
    DirectoryExistEmpty = 1
    DirectoryExistNotEmpty = 2
    MissingParent = 3
    DirectoryNotExist = 4
    FileNotExist = 5


def check_path_existence(
    path_str: str,
    path_type: Literal['file', 'dir', 'auto'] = 'auto',
) -> Existence:
    """Check whether a file or a directory exists at the expected path.

    Args:
        path_str (str):
            Path to check.
        path_type (Literal[, optional):
            What kind of file do we expect at the path.
            Choose among `file`, `dir`, `auto`.
            Defaults to 'auto'.    path_type = path_type.lower()

    Raises:
        KeyError: if `path_type` conflicts with `path_str`

    Returns:
        Existence:
            0. FileExist: file at path_str exists.
            1. DirectoryExistEmpty: folder at path exists and.
            2. DirectoryExistNotEmpty: folder at path_str exists and not empty.
            3. MissingParent: its parent doesn't exist.
            4. DirectoryNotExist: expect a folder at path_str, but not found.
            5. FileNotExist: expect a file at path_str, but not found.
    """
    path_type = path_type.lower()
    assert path_type in {'file', 'dir', 'auto'}
    pathinfo = Path(path_str)
    if not pathinfo.parent.is_dir():
        return Existence.MissingParent
    suffix = pathinfo.suffix.lower()
    if path_type == 'dir' or\
            path_type == 'auto' and suffix == '':
        if pathinfo.is_dir():
            if len(os.listdir(path_str)) == 0:
                return Existence.DirectoryExistEmpty
            else:
                return Existence.DirectoryExistNotEmpty
        else:
            return Existence.DirectoryNotExist
    elif path_type == 'file' or\
            path_type == 'auto' and suffix != '':
        if pathinfo.is_file():
            return Existence.FileExist
        elif pathinfo.is_dir():
            if len(os.listdir(path_str)) == 0:
                return Existence.DirectoryExistEmpty
            else:
                return Existence.DirectoryExistNotEmpty
        if path_str.endswith('/'):
            return Existence.DirectoryNotExist
        else:
            return Existence.FileNotExist


def prepare_output_path(output_path: str,
                        allowed_suffix: List[str] = [],
                        tag: str = 'output file',
                        path_type: Literal['file', 'dir', 'auto'] = 'auto',
                        overwrite: bool = True) -> None:
    """Check output folder or file.

    Args:
        output_path (str): could be folder or file.
        allowed_suffix (List[str], optional):
            Check the suffix of `output_path`. If folder, should be [] or [''].
            If could both be folder or file, should be [suffixs..., ''].
            Defaults to [].
        tag (str, optional): The `string` tag to specify the output type.
            Defaults to 'output file'.
        path_type (Literal[, optional):
            Choose `file` for file and `dir` for folder.
            Choose `auto` if allowed to be both.
            Defaults to 'auto'.
        overwrite (bool, optional):
            Whether overwrite the existing file or folder.
            Defaults to True.

    Raises:
        FileNotFoundError: suffix does not match.
        FileExistsError: file or folder already exists and `overwrite` is
            False.

    Returns:
        None
    """
    if path_type.lower() == 'dir':
        allowed_suffix = []
    exist_result = check_path_existence(output_path, path_type=path_type)
    if exist_result == Existence.MissingParent:
        warnings.warn(
            f'The parent folder of {tag} does not exist: {output_path},' +
            f' will make dir {Path(output_path).parent.absolute().__str__()}')
        os.makedirs(
            Path(output_path).parent.absolute().__str__(), exist_ok=True)

    elif exist_result == Existence.DirectoryNotExist:
        os.mkdir(output_path)
        print(f'Making directory {output_path} for saving results.')
    elif exist_result == Existence.FileNotExist:
        suffix_matched = \
            check_path_suffix(output_path, allowed_suffix=allowed_suffix)
        if not suffix_matched:
            raise FileNotFoundError(
                f'The {tag} should be {", ".join(allowed_suffix)}: '
                f'{output_path}.')
    elif exist_result == Existence.FileExist:
        if not overwrite:
            raise FileExistsError(
                f'{output_path} exists (set overwrite = True to overwrite).')
        else:
            print(f'Overwriting {output_path}.')
    elif exist_result == Existence.DirectoryExistEmpty:
        pass
    elif exist_result == Existence.DirectoryExistNotEmpty:
        if not overwrite:
            raise FileExistsError(
                f'{output_path} is not empty (set overwrite = '
                'True to overwrite the files).')
        else:
            print(f'Overwriting {output_path} and its files.')
    else:
        raise FileNotFoundError(f'No Existence type for {output_path}.')


def check_input_path(
    input_path: str,
    allowed_suffix: List[str] = [],
    tag: str = 'input file',
    path_type: Literal['file', 'dir', 'auto'] = 'auto',
):
    """Check input folder or file.

    Args:
        input_path (str): input folder or file path.
        allowed_suffix (List[str], optional):
            Check the suffix of `input_path`. If folder, should be [] or [''].
            If could both be folder or file, should be [suffixs..., ''].
            Defaults to [].
        tag (str, optional): The `string` tag to specify the output type.
            Defaults to 'output file'.
        path_type (Literal[, optional):
            Choose `file` for file and `directory` for folder.
            Choose `auto` if allowed to be both.
            Defaults to 'auto'.

    Raises:
        FileNotFoundError: file does not exists or suffix does not match.

    Returns:
        None
    """
    if path_type.lower() == 'dir':
        allowed_suffix = []
    exist_result = check_path_existence(input_path, path_type=path_type)

    if exist_result in [
            Existence.FileExist, Existence.DirectoryExistEmpty,
            Existence.DirectoryExistNotEmpty
    ]:
        suffix_matched = \
            check_path_suffix(input_path, allowed_suffix=allowed_suffix)
        if not suffix_matched:
            raise FileNotFoundError(
                f'The {tag} should be {", ".join(allowed_suffix)}:' +
                f'{input_path}.')
    else:
        raise FileNotFoundError(f'The {tag} does not exist: {input_path}.')

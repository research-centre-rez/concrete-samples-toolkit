import os

def append_file_extension(filename:str, extension:str):
    '''
    Appends a file extension to `filename`. Works with both `.extenstion` and `extension` without '.'
    Args:
        filename (str): Base filename, can contain file extension (in case you want to change a file extension)
        extension (str): Extension, will be appended to filename
    Returns:
        new_filename (str)
    '''

    path, name = os.path.split(filename)
    base, _    = os.path.splitext(name)
    ext        = extension if extension.startswith('.') else f'.{extension}'

    return os.path.join(path, f'{base}{ext}')

def create_out_filename(filename:str, prefixes:list[str], suffixes:list[str]):
    '''
    Add suffixes to filename, used when saving a file into storage.
    Args:
        filename (str): Base filename, can contain the file extension and relative path
        prefixes (list[str]): Prefixes to be added before the filename
        suffixes (list[str]): Suffixes to be added after the filename
    Returns:
        new_filename (str): A new filename
    '''

    # Removes the file extension (works even if there is none present)
    # Removes relative path
    path, name = os.path.split(filename)
    base, _    = os.path.splitext(name)

    suff = '_'.join(suffixes)
    pref = '_'.join(prefixes)

    parts = [part for part in (pref, base, suff) if part]
    new_f = '_'.join(parts)

    return os.path.join(path, new_f)

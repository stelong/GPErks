from pathlib import Path


def posix_path(*args: str) -> str:
    return Path().joinpath(*args).as_posix()

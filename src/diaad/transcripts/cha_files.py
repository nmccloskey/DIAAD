import random
from pathlib import Path

import pylangacq
from tqdm import tqdm
from psair.core.logger import logger, get_rel_path


def truncate_path_to_input_dir(path: str | Path, input_dir: str | Path) -> str:
    """
    Return a portable path that starts at the configured input directory.
    """
    path = Path(path).expanduser().resolve()
    input_dir = Path(input_dir).expanduser().resolve()

    try:
        rel_path = path.relative_to(input_dir)
    except ValueError:
        logger.warning(
            "CHAT file %s is outside configured input directory %s; using filename only.",
            get_rel_path(path),
            get_rel_path(input_dir),
        )
        return path.name

    return Path(input_dir.name, rel_path).as_posix()


def read_cha_files(input_dir, shuffle=False):
    """
    Read CHAT (.cha) files from the given input directory and return
    a dict of {input-relative path: pylangacq.Reader} objects.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    cha_files = list(input_dir.rglob("*.cha"))

    if shuffle:
        logger.info("Shuffling the list of .cha files.")
        random.shuffle(cha_files)

    logger.info(f"Reading .cha files from directory: {get_rel_path(input_dir)}")
    chats = {}

    for cha in tqdm(cha_files, desc="Reading .cha files..."):
        try:
            chat_data = pylangacq.Reader.from_files([str(cha)], parallel=False)
            chats[truncate_path_to_input_dir(cha, input_dir)] = chat_data
        except Exception as e:
            logger.error(f"Failed to read {get_rel_path(cha)}: {e}")

    logger.info(f"Successfully read {len(chats)} .cha files from {get_rel_path(input_dir)}.")
    return chats

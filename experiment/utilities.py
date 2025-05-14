# Taken from the BNL AQNSim Repo
from datetime import datetime
import json
import os
import re
import uuid

def list_json_files_in_folder(folder_path):
    """
    Get list of all json files in a folder
    """
    json_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.json')
    ]
    return json_files


def datetime_now_string() -> str:
    """
    Returns the current UTC datetime in a string format as ``YYYY-mm-ddTHH:MM:SS.ssssssssssZ``.
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%fZ")


def uuid_string() -> str:
    """
    Returns a unique unguessable identifier (UUID) string.
    """
    return str(uuid.uuid4())


def make_dir(filepath: str) -> str:
    """
    Creates a folder if it does not already exist.

    :param filepath: The filepath to the directory to make.

    :returns: The created filepath.
    """
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    return filepath


def read_json_files(filepath: str, regex: str, num_files: int = None) -> list[dict]:
    """Reads all json files that match the ``regex`` in the
    directory specified by ``path`` where  the file extension is
    assumed to be ``".json"``

    :param filepath: The filepath to search.
    :param regex: A regex string to match should start as ``r"..."``.

    :returns: A list of collected JSON data.
    """
    filenames = [
        os.path.join(filepath, f)
        for f in os.listdir(filepath)[0:num_files]
        if (
            f.endswith(".json")
            and os.path.isfile(os.path.join(filepath, f))
            and bool(re.match(regex, f))
        )
    ]

    return [read_json(filename) for filename in filenames]


def write_json(json_dict: dict, filename: str = None):
    """Writes the dictionary to JSON file with name ``filename``.

    :param json_dict: The dictionary to write as a JSON file.
    :param filename: The name of the JSON file. Note that ``.json`` extension is automatically added.

    :returns: ``None``
    """
    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_dict, indent=2))


def read_json(filename: str) -> dict:
    """Reads data from a JSON file.

    :param filename: The path to the JSON file. Note this string must contain the ``.json`` extension.

    :returns: The dictionary read from the JSON file.
    """

    with open(filename) as file:
        json_dict = json.load(file)

    return json_dict

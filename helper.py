import os
import site
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    added = set()

    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"

        if not nvidia_dir.exists():
            continue

        for dll in nvidia_dir.rglob("*.dll"):
            dll_dir = str(dll.parent)

            if dll_dir not in added:
                os.add_dll_directory(dll_dir)
                added.add(dll_dir)
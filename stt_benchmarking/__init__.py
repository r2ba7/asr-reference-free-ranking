import glob
import os
import sys


def _register_ffmpeg_dll_dirs() -> None:
    """
    On Windows, Python does not search PATH for DLL dependencies, so torchcodec
    (used by torchaudio.load) cannot find the FFmpeg shared libraries even when
    they are on PATH. Register every PATH folder that contains them explicitly.
    """
    if sys.platform != "win32":
        return
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if folder and glob.glob(os.path.join(folder, "avcodec-*.dll")):
            try:
                os.add_dll_directory(folder)
            except OSError:
                pass


_register_ffmpeg_dll_dirs()

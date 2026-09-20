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


def _patch_torchaudio_for_speechbrain() -> None:
    """
    speechbrain 0.5.x calls torchaudio.set_audio_backend("soundfile") on Windows,
    but torchaudio >= 2.9 removed it. Provide a no-op so speechbrain imports.
    """
    try:
        import torchaudio
    except ImportError:
        return
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None


def _patch_speechbrain_lazy_modules() -> None:
    """
    speechbrain >= 1.0 lazy-loads optional integrations (e.g. k2) and tries to
    stop `inspect` from triggering those imports by checking for "/inspect.py",
    which never matches on Windows paths. Answer `__dunder__` lookups on a
    not-yet-loaded lazy module with AttributeError instead of importing it.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        return
    original_getattr = LazyModule.__getattr__

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__") and self.lazy_module is None:
            raise AttributeError(attr)
        return original_getattr(self, attr)

    LazyModule.__getattr__ = __getattr__


_register_ffmpeg_dll_dirs()
_patch_torchaudio_for_speechbrain()
_patch_speechbrain_lazy_modules()

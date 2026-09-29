# STT Benchmarking

Speech-to-text benchmarking, deterministic consensus fusion with a ROVER baseline, and reference-free evaluation for Arabic and English. An optional LLM refinement stage exists in the pipeline but is disabled in every run; no published result depends on it. Managed with [uv](https://docs.astral.sh/uv/); Python 3.10.

The code lives in the `stt_benchmarking/` package (flat layout, installed in editable mode).

## RDI corpus and substitution package

The dictionary substitution used for RDI is part of the RDI dataset rather than of this
code, and is released with the dataset. The RDI corpus is not currently distributed, so the
package is not either; if the dataset is released, the package goes with it.

Common Voice Arabic and LibriSpeech reproduce from a clean install. RDI results can be
audited against the released outputs but not regenerated: calling the fusion with
`substitute=True` requires that package and raises otherwise.

## Environments

The models cannot share one environment because their requirements conflict:

- The Cohere models need `transformers >= 5.4`, while NeMo 2.3.2 needs `transformers <= 4.52`.
- Two Arabic models ship a speechbrain 0.5 config, while the other speechbrain models
  need speechbrain 1.x.

So `pyproject.toml` defines three mutually exclusive dependency groups, each installed into
its own virtual environment. `uv.lock` covers all three.

| Environment      | Group    | Key versions                                          | Runs                                                                                         |
| ---------------- | -------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `.venv`          | `main`   | transformers 4.51 to 4.52, NeMo 2.3.2, speechbrain 1.1.0 | All models except the ones below, including the English speechbrain model                    |
| `.venv-hubert`   | `hubert` | transformers 4.57.1, speechbrain 0.5.16               | Arabic `speechbrain.py` and Arabic `hubert.py`                                               |
| `.venv-cohere`   | `cohere` | transformers >= 5.4, huggingface_hub >= 1.0           | `arabic/cohere_arabic.py` and `english/cohere_english.py`                                    |

All three also install the `dev` group (ipykernel, ipywidgets, nbconvert). Shared packages
(torch 2.9.0 CUDA 12.8, pandas, datasets, ...) are in the base dependencies.

`english/owsm.py` needs `espnet`, which is not installed in any environment.

## Setup

### Prerequisites (Windows)

1. **uv**: `winget install --id=astral-sh.uv -e`
2. **NVIDIA driver** new enough for CUDA 12.8. The PyTorch wheels bundle their own CUDA
   runtime, so the CUDA Toolkit is not required.
3. **FFmpeg 7.x, shared build** on `PATH`, for `torchaudio.load` (via `torchcodec`):
   `winget install Gyan.FFmpeg.Shared --version 7.1.1`. FFmpeg 9 does not work with
   torchcodec 0.8, and the "essentials" build has no DLLs.
4. **Developer Mode** (Settings, For developers): speechbrain 0.5 creates symlinks when it
   downloads models, which needs it on Windows.

### Create the environments

Main environment (`.venv`, the default):

```
uv sync
```

The other two use `UV_PROJECT_ENVIRONMENT` to install into a separate folder. Run the three
lines together in the same window, and clear the variable afterwards so the next `uv sync`
goes back to `.venv`.

cmd:

```
set UV_PROJECT_ENVIRONMENT=.venv-hubert
uv sync --no-default-groups --group dev --group hubert
set UV_PROJECT_ENVIRONMENT=

set UV_PROJECT_ENVIRONMENT=.venv-cohere
uv sync --no-default-groups --group dev --group cohere
set UV_PROJECT_ENVIRONMENT=
```

PowerShell:

```
$env:UV_PROJECT_ENVIRONMENT = ".venv-hubert"
uv sync --no-default-groups --group dev --group hubert
$env:UV_PROJECT_ENVIRONMENT = $null
```

Close notebook kernels that use an environment before syncing it; otherwise Windows refuses
to replace DLLs that are in use.

### Adding a package

`uv add <package>` adds it to the base dependencies. To add it to one group only, use
`uv add --group main <package>` (or `hubert`, `cohere`). Run it with
`UV_PROJECT_ENVIRONMENT` set if you want that environment updated too.

## Running

Scripts: `uv run python script.py` (uses `.venv`). For another environment, set
`UV_PROJECT_ENVIRONMENT` first, or call `.venv-cohere\Scripts\python.exe` directly.

Notebooks: choose the interpreter that matches the model you are running
(`.venv`, `.venv-hubert` or `.venv-cohere`) with "Select Kernel" in VS Code.

The Cohere Transcribe repositories on Hugging Face may require accepting the model license
and logging in (`hf auth login`).

## Compatibility shims

`stt_benchmarking/__init__.py` applies three Windows/version workarounds when the package
is imported. Import `stt_benchmarking` before `torchaudio` or `speechbrain`.

- Registers the FFmpeg `bin` folder found on `PATH` with `os.add_dll_directory`, so
  torchcodec can load the FFmpeg DLLs.
- Adds a no-op `torchaudio.set_audio_backend`, which speechbrain 0.5 calls and torchaudio
  2.9 removed.
- Stops speechbrain 1.x's lazy imports from being triggered by `inspect` on Windows, which
  otherwise fails with `No module named 'k2'`.

## Troubleshooting

| Symptom                                                           | Cause and fix                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `Could not load libtorchcodec`                                    | Install FFmpeg 7.x shared (prerequisite 3); import `stt_benchmarking` first |
| `WinError 1314 ... required privilege` when loading speechbrain   | Enable Developer Mode (prerequisite 4)                                    |
| `No module named 'speechbrain.inference'`                         | Wrong environment: use `.venv` (speechbrain 1.x) for those models         |
| `There is no such class as ...HuggingFaceWav2Vec2`                | Wrong environment: use `.venv-hubert`                                     |
| `cannot import name 'CohereAsrForConditionalGeneration'`          | Wrong environment: use `.venv-cohere`                                     |
| `Access is denied` while running `uv sync`                        | A kernel or process is using that environment; close it and retry         |

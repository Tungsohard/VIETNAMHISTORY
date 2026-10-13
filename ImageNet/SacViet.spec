# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

project = Path(SPECPATH)
datas = []
binaries = []
hiddenimports = []

for package in ("gradio", "gradio_client", "safehttpx", "groovy", "webview", "clr_loader", "pythonnet"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("basicsr", filter=lambda name: ".test" not in name)
datas += [
    (str(project / "third_party" / "DDColor"), "third_party/DDColor"),
]

a = Analysis(
    [str(project / "windows_launcher.py")],
    pathex=[str(project), str(project / "third_party" / "DDColor")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["jupyter", "notebook", "matplotlib.tests", "numpy.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SacViet",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="SacViet",
)

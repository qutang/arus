# -*- mode: python ; coding: utf-8 -*-

import platform
import sys

sys.setrecursionlimit(5000)
block_cipher = None

if platform.system() == 'Windows':
    binaries = [("../../.venv/Lib/site-packages/mbientlab/metawear/MetaWear.Win32.dll", "."),
                ("../..//.venv/Lib/site-packages/mbientlab/warble/warble.dll", ".")]
elif platform.system() == 'Linux':
    binaries = [("../../.venv/lib/python3.7/site-packages/mbientlab/metawear/libmetawear.so", "."),
                ("../..//.venv/lib/python3.7/site-packages/mbientlab/warble/libwarble.so", ".")]

a = Analysis(['main.py'],
             pathex=['.', '../../.venv/Lib/site-packages/scipy/.libs'],
             binaries=binaries,
             datas=[("../../data/single/mhealth/feature/multi_placements/muss.feature.csv.gz", "./data/single/mhealth/feature/multi_placements/"),
                    ("../../data/single/mhealth/class_labels/multi_tasks/muss.class.csv", "./data/single/mhealth/class_labels/multi_tasks/"), ("./assets/dom_wrist.png", "./assets/"), ("./assets/right_ankle.png", "./assets/"), ("./assets/switch_now.mp3", "./assets/"), ("./assets/keep_going.mp3", "./assets/"), ("./README.md", ".")],
             hiddenimports=['scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.integrate._dop',
                            'scipy.integrate.lsoda',
                            'tkinter',
                            'sklearn.utils._cython_blas',
                            'sklearn.utils.sparsetools._graph_validation',
                            'sklearn.utils.sparsetools._graph_tools',
                            'sklearn.utils.lgamma'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='arus_demo',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='arus_demo')

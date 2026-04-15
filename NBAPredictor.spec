# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

nba_datas, nba_binaries, nba_hidden = collect_all('nba_api')
scraper_datas, scraper_binaries, scraper_hidden = collect_all('cloudscraper')
lgbm_datas, lgbm_binaries, lgbm_hidden = collect_all('lightgbm')
xgb_datas, xgb_binaries, xgb_hidden = collect_all('xgboost')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=nba_binaries + scraper_binaries + lgbm_binaries + xgb_binaries,
    datas=[
        ('web/templates', 'web/templates'),
    ] + nba_datas + scraper_datas + lgbm_datas + xgb_datas + collect_data_files('certifi'),
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils.murmurhash',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree._utils',
        'sklearn.ensemble._forest',
        'sklearn.ensemble._gb_losses',
        'sklearn.linear_model._logistic',
        'xgboost',
        'xgboost.sklearn',
        'xgboost.core',
        'joblib',
        'lxml',
        'lxml.etree',
        'html5lib',
        'requests',
        'certifi',
        'charset_normalizer',
        'idna',
        'urllib3',
        'flask',
        'jinja2',
        'markupsafe',
        'werkzeug',
        'itsdangerous',
        'click',
        'lightgbm',
        'lightgbm.sklearn',
        'lightgbm.basic',
    ] + nba_hidden + scraper_hidden + lgbm_hidden + xgb_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'wx', 'IPython', 'jupyter'],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NBAPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NBAPredictor',
)

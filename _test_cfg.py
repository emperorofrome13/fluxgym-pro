import sys, os
sys.path.insert(0, '.')
from app import build_cfg
try:
    cfg = build_cfg('Krea 2 (Raw)', 'test_lora', 'test_lora_v1', 'E:/aiprojects/musubi-tuner', 0.0001, 16, 16, 1024, 1, 1, 1, 2, 'bf16', '16 GB (balanced)', 'Automatic (recommended)', 12, False, 1, 'adamw8bit', False, 'constant', 0, 1.0, '', 2, True, 'bf16', True, False, 'Model default (recommended)', 0, 0, 1, 0, False, '', 0)
    print('CONFIG OK, output_dir=', cfg.get('output_dir'))
except Exception as e:
    print('CONFIG ERR:', type(e).__name__, e)

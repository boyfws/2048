import random
import numpy as np
import torch
import os

def seed_all(seed=42, deterministic=True):
    """
    Фиксирует сиды для всех основных библиотек для воспроизводимости результатов.
    
    Args:
        seed (int): Значение seed для фиксации случайности (по умолчанию 42)
        deterministic (bool): Использовать детерминистические алгоритмы где возможно
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # для multi-GPU
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        


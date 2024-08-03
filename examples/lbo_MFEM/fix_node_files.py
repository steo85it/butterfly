#!/usr/bin/env python

import numpy as np
import sys

from pathlib import Path

path = Path(sys.argv[1])

np.fromfile(path, dtype=np.float64).reshape(3, -1).T.tofile(path)

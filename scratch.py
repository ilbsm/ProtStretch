import pandas as pd
import os
from Tools import *

data = load_data(os.path.join(os.path.dirname(__file__), "data/", "aa01_inverse.afm"), inverse=True)

print(data)

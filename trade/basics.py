# from typing import Union 
from itertools import product # 笛卡儿积 
from toolz.functoolz import compose # 函数复合

from pynverse import inversefunc 
from pynverse import piecewise
# inversefunc, 超爱这个功能
# fi = inversefunc(np.exp,domain=(-np.inf,+np.inf),image=(0,+np.inf),open_domain=True)
# pw = lambda x: piecewise(x,[x<1,(x>=1)*(x<3),x>=3],[lambda x: x, lambda x: x**2, lambda x: x+6])
'''
| 参数 | 含义 |
| --- | --- |
|`func` | 比如np.cos |
|`domain` | np.cos的domain |
| `image` | np.cos的image |
| `open_domain ` | 是否为开区间 |
'''


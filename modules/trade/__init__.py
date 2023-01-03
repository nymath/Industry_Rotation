import os
__version__ = "0.1"

rawpath = f'{os.getenv("HOME")}/.rqalpha/bundle'

if not os.path.exists(rawpath):
    print('数据源路径不存在, 请输入rqalpha download-bundle安装')
    

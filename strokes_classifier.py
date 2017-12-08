
# coding: utf-8

# In[2]:

"""
SaikyoABC.pyのプログラムで取得したstrokes.csvファイルの各ストロークの種類（補助線か？計算式か？）を分類します。
"""
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import SaikyoABC as sk
import csv
from PIL import Image, ImageDraw
print (sk.__version__)
import sys


# In[3]:

class strokes_classifier(object):
    def __init__(self,strokes):
        self.strokes = strokes
        
    """
    以下の関数でストロークのnumpy行列を入力として受け取り、それらをストロークのブロックにクラスタリングします。
    """
    def strokes_split(strokes):
        
        
    
    


# In[9]:

abc = sk.SaikyoABC("strokes.csv")
print(abc.stroke)
print(abc.stroke.STROKE)
for img in abc.stroke.STROKE:
    plt.plot(img[0],img[1])


# In[ ]:




# In[ ]:




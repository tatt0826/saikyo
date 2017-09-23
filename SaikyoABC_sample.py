
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import SaikyoABC as sk
print (sk.__version__)


# In[2]:

abc = sk.SaikyoABC("strokes.csv")
abc.stroke.ix[10]


# In[3]:

s = abc.stroke.STROKE[10]
plt.plot(s[0],s[1])


# In[4]:

# PENTYPEがSTYLUSのものだけについて、ストローク長を計算する
lengths = [sk.strokeLength(s) for s,p in zip(abc.stroke.STROKE,abc.stroke.PENTYPE) if p == "STYLUS"]


# In[5]:

plt.plot(lengths)


# In[6]:

import PIL
image = PIL.Image.open("image.gif")
overlayimage = abc.overlay_image(image)
overlayimage


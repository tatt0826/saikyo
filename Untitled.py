
# coding: utf-8

# In[42]:

get_ipython().magic(u'pylab inline')
from SaikyoABC import SaikyoABC

class A(SaikyoABC):
    def __init__(self,filename):
        abc = SaikyoABC(filename)
    print()


# In[ ]:




# In[ ]:




# In[40]:

filenum = "4"
abc = A("strokes" + filenum + ".csv")
abc.stroke_classifier()


# In[36]:

import PIL
imagename = "image" + filenum + ".gif"
image = PIL.Image.open(imagename)
overlayimage = abc.overlay_image(image)
overlayimage2 = abc.overlay_block_image(overlayimage)
overlayimage2


# In[ ]:




# In[ ]:




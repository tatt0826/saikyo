
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import SaikyoABC as sk
print (sk.__version__)



# In[ ]:


# In[2]:

abc = sk.SaikyoABC("strokes.csv")
abc.stroke.ix[10]




# In[40]:

# In[3]:
for n in [25,30]:
    s = abc.stroke.STROKE[n]
    plt.plot(s[0],s[1])
"""    
s = abc.stroke.STROKE[40]
plt.plot(s[0],s[1])
"""




# In[41]:

# In[4]:

# PENTYPEがSTYLUSのものだけについて、ストローク長を計算する
lengths = [sk.strokeLength(s) for s,p in zip(abc.stroke.STROKE,abc.stroke.PENTYPE) if p == "STYLUS"]
print(lengths)



# In[42]:


# In[5]:

plt.plot(lengths)




# In[43]:

# In[6]:

import PIL
image = PIL.Image.open("image.gif")
overlayimage = abc.overlay_image(image)
overlayimage


# In[ ]:

def get_time_lag(abc):
    t_length = len(abc.stroke.Timestamp) - 1
    for i in range(t_length):
        print(abc.stroke.Timestamp[i+1]-abc.stroke.Timestamp[i])
get_time_lag(abc)


# In[ ]:

stroke_scale = np.array([[np.min(abc.stroke.STROKE[i][0]),
                          np.max(abc.stroke.STROKE[i][0]),
                          np.min(abc.stroke.STROKE[i][1]),
                          np.max(abc.stroke.STROKE[i][1])] 
                         for i in range(len(abc.stroke))])
for i in range(len(abc.stroke)):
    print(np.min(abc.stroke.STROKE[i][0]))
print(stroke_scale)


# In[ ]:

print(abc.stroke.STROKE[30][0][0])


# In[ ]:

print(abc.stroke.STROKE[30][1][-1])
print(abc.stroke.STROKE[30])
print(abc.stroke.STROKE[31])


# In[ ]:

np.append(abc.stroke.STROKE[1],[2])
print(abc.stroke.STROKE[1])


# In[ ]:

a = np.array([[3,4],[5]])
print(type(a[0]))
#print(a)
a[0] = np.append(a[0],[1])
#print(a)
print(type(a[0]))


# In[ ]:

a = []
if len(a) == 0 :
    print("a")


# In[30]:

a = [[1]]
a.append(1)
print (a)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




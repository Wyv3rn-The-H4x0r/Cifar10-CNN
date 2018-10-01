
# coding: utf-8

# In[1]:


# Model Predict and analyte :


# In[2]:


from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import keras.models as models
import numpy as np


# In[5]:


filename = "YourKerasModel.h5"
model = models.load_model(filename)


# In[6]:


# Load Image for Predict | Choose to your Own Picture
img = Image.open("ChangeMe.jpg")


# In[7]:


size = 100
step_size = 50
cars = []
cars_drawn = []


# In[8]:


for x in range(0,img.size[0] - size, step_size):
    for y in range(0, img.size[1] - size, step_size):
        part = img.crop((x, y, x + size, y + size))
        data = np.asarray(part.resize((32,32), resample=Image.BICUBIC))
        date = data.astype(np.float32) / 255.
    
        pred = model.predict(data.reshape(-1, 32 ,32, 3))
        if pred[0][0] > 0.97:
            cars.append((x , y))

out = img.copy()
draw = ImageDraw.Draw(out)

for car in cars:
    exists = False
    for car_drawn in cars_drawn:
        if car[0] >= car_drawn[0] and car[0] <= car_drawn[0] + size:
            if car[1] >= car_drawn[1] and car[1] <= car_drawn[1] + size:
                exists = True
                
    if exists == False: 
        points = (
            car,
            (car[0], car[1] + size),
            (car[0] + size, car[1] + size),
            (car[0] + size, car[1]), car)

        draw.line(points, "yellow", 2)
        cars_drawn.append(car)
    


# In[11]:


print("[Saved]Picture saved as -> Output.jpg")
out.save("Output.jpg", quallty=90)
plt.imshow(out)
plt.show()


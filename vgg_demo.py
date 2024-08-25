import numpy as np
from PIL import Image
import samsara
from samsara.models import VGG16
import cv2


file_name = 'zebra.jpg'
img_path = samsara.utils.get_local_file(file_name)
img = Image.open(img_path)

x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=False)

dir(model)
with samsara.test_mode():
    with samsara.using_config('visualize_forward', False):
        y = model(x)
    z = y.backward(visualize=False)
predict_id = np.argmax(y.data)

# model.plot(x, to_file='vgg.pdf')
labels = samsara.datasets.ImageNet.labels()
print(labels[predict_id])
cv2.waitKey(0)
cv2.destroyAllWindows()
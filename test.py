from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import transforms
import sys
from myCNN import MyCNN

classes = ['gossiping', 'isolation', 'laughing',
           'pullinghair', 'punching', 'quarrel', 
           'slapping', 'stabbing', 'strangle',
           'unbulloying']

image_loader = transforms.Compose([
    transforms.RandomResizedCrop(224,224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# This method only work in CUDA library
def load_image(image_loader, name):
    images = Image.open(name)
    images = image_loader(images).float()
    images = Variable(images, requires_grad = True)
    images = image.unsqueeze(0)
    return images.cuda()


#test_model = MyCNN()

# if torch.cuda.device_count() > 1:
#     test_model = nn.DataParallel(test_model()).cuda()
# else:
#     test_model = test_model.cuda()

test_model = nn.DataParallel(test_model()).cuda()
test_model.load_state_dict(torch.load('./mycnn.ckpt'))

test_model.eval()

image = load_image(image_loader, sys.argv[1])

outputs = test_model(image)

_,predicted = torch.max(outputs.data,1)

result = predicted.item()

print(classes[result])


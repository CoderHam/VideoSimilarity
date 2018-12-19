import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

model_res18 = models.resnet18(pretrained=True).cuda()
model_res18.classifier = model_res18._modules.get('avgpool')
model_res18 = nn.Sequential(*list(model_res18.classifier.children())[:-1])

def get_vector_resnet18(img_path):
    img = Image.open(img_path)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    output = model_res18(t_img)
    feature_vector = output.data.cpu()
    return feature_vector

# get_vector_resnet18('data/images/golden1.jpg')

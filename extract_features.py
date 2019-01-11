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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_res50 = models.resnet50(pretrained=True).cuda()
for param in model_res50.parameters():
    param.requires_grad = False
# model_res50.classifier = model_res50._modules.get('avgpool')
# model_res50.classifier = res50_classifier
res50_classifier = nn.Sequential(*list(model_res50.children())[:-1])

def get_vector_resnet50(img_path):
    img = Image.open(img_path)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()
    output = res50_classifier(t_img)
    feature_vector = output.data.cpu()
    return feature_vector.numpy().flatten()

# print(get_vector_resnet50('data/images/golden1.jpg')

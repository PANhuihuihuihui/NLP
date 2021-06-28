from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device)

# one method
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
# seconde method
summary(model,(1,3,80,1200))
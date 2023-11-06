import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json

def main():

  # Create Parse using ArgumentParser
  parser = argparse.ArgumentParser()

  # Create command line arguments using add_argument() from ArguementParser method
  parser.add_argument('image', type=str, default='flower_data/test/21/image_06805.jpg', help='predicted image path')
  parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='checkpoint for pre-trained model')
  parser.add_argument('--top_k', type=int, default=5, help='top k class')
  parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='categories name of flowers species')
  parser.add_argument('--gpu', type=str, default='gpu', help='Change to process by GPU')

  # Return parse_args() data structure that stores the command line arguments object
  in_arg = parser.parse_args()

  with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
  predictions_count = len(cat_to_name)

  def create_model(arch, lyr2, learning_rate):
    model = getattr(models, arch)(pretrained=True)

    if 'vgg' in arch:
      lyr1 = 25088
    elif 'alexnet' in arch:
      lyr1 = 9216
    elif 'densenet' in arch:
      lyr1 = 1024
    elif 'efficientnet' in arch:
      lyr1 = 2304
    elif 'mnasnet' in arch or 'mobilenet' in arch:
      lyr1 = 1280
      
    model.classifier = nn.Sequential(nn.Linear(lyr1, lyr2),
                                 nn.ReLU(),
                                 nn.Dropout(0.05),
                                 nn.Linear(lyr2, predictions_count),
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, optimizer

  #Load Check Point
  def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model, optimizer = create_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['lr'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return model, optimizer

  # To Process Images
  def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_img = Image.open(image)

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    np_image = preprocess(pil_img)
    return np_image

  # Predict Function
  def predict(image_path=in_arg.image, model=in_arg.checkpoint, device=in_arg.gpu, topk=in_arg.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model, optimizer = load_checkpoint(model)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
    model.to(device)
    model.eval()
    np_image = process_image(image_path)

    with torch.no_grad():
        np_image = np_image.to(device)
        np_image.unsqueeze_(0)
        np_image.float()
        logps = model.forward(np_image)
        ps = torch.exp(logps)
        top_ps, top_class_idx = ps.topk(topk, dim=1)
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    probs = top_ps.cpu().numpy()[0]
    classes = [class_to_idx_inverted[i] for i in top_class_idx.cpu().numpy()[0]]

    return probs, classes

  probs, classes = predict()
  flower_classes = [cat_to_name[i] for i in classes]
  print(flower_classes)
  print(probs)


if __name__ == '__main__':
  main()
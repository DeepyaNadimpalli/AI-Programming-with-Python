import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json

def main():
  # Create Parse using ArgumentParser
  parser = argparse.ArgumentParser()

  # Create command line arguments using add_argument() from ArguementParser method
  parser.add_argument('data_dir', type=str, default='flower_data', help='data set path')
  parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='saved checkpoint path')
  parser.add_argument('--arch', type=str, default='vgg16', help='CNN architecture')
  parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
  parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units')
  parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
  parser.add_argument('--gpu', type=str, default='gpu', help='Change to process by GPU')

  # Return parse_args() data structure that stores the command line arguments object

  in_arg = parser.parse_args()

  #Load Data
  data_dir = in_arg.data_dir
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'
  # Define your transforms for the training, validation, and testing sets
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

  # Load the datasets with ImageFolder
  train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
  valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
  test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

  # Using the image datasets and the trainforms, define the dataloaders
  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
  validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
  testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)


  #Train Data
  learning_rate=in_arg.learning_rate
  arch=in_arg.arch
  epochs=in_arg.epochs
  device=in_arg.gpu

  with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

  predictions_count = len(cat_to_name)
  device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
  model = getattr(models, arch)(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False
  steps = 0
  running_loss = 0
  print_every=20
  
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

  lyr2 = in_arg.hidden_units
  model.classifier = nn.Sequential(nn.Linear(lyr1, lyr2),
                                 nn.ReLU(),
                                 nn.Dropout(0.05),
                                 nn.Linear(lyr2, predictions_count),
                                 nn.LogSoftmax(dim=1))
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
  model.to(device);

  for epoch in range(epochs):
    for inputs, labels in trainloader:
      steps += 1
      # Move input and label tensors to the default device
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if steps % print_every == 0:
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
          for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/print_every:.3f}.. "
            f"Test loss: {test_loss/len(validloader):.3f}.. "
            f"Test accuracy: {accuracy/len(validloader):.3f}")
        running_loss = 0
        model.train()

  # Save Check Point
  model.class_to_idx = train_dataset.class_to_idx
  
  checkpoint = {'optimizer_dict': optimizer.state_dict(),
                'arch':in_arg.arch,
                'hidden_units': lyr2,
                'lr': learning_rate,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()}

  torch.save(checkpoint, in_arg.save_dir)

  print("====Training completed.====")


if __name__ == '__main__':
    main()
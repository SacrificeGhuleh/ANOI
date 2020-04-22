import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow  # as pyplot_imshow


def my_imshow(img):
  npimg = img.numpy()
  # npimg = np.array(img)
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


class LeNet(torch.nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
    # Max-pooling
    self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
    # Convolution
    self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
    # Max-pooling
    self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
    # Fully connected layer
    self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
    self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
    self.fc3 = torch.nn.Linear(84, 10)  # convert matrix with 84 features to a matrix of 10 features (columns)

  def forward(self, x):
    # convolve, then perform ReLU non-linearity
    x = torch.nn.functional.relu(self.conv1(x))
    # max-pooling with 2x2 grid
    x = self.max_pool_1(x)
    # convolve, then perform ReLU non-linearity
    x = torch.nn.functional.relu(self.conv2(x))
    # max-pooling with 2x2 grid
    x = self.max_pool_2(x)
    # first flatten 'max_pool_2_out' to contain 16*5*5 columns
    # read through https://stackoverflow.com/a/42482819/7551231
    x = x.view(-1, 16 * 5 * 5)
    # FC-1, then perform ReLU non-linearity
    x = torch.nn.functional.relu(self.fc1(x))
    # FC-2, then perform ReLU non-linearity
    x = torch.nn.functional.relu(self.fc2(x))
    # FC-3
    x = self.fc3(x)

    return x


def train(data, model):
  model.train()

  learning_rate = 0.01
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  num_epochs = 5
  p = 1
  with open("loss.txt", "wt") as f:
    for epoch in range(num_epochs):
      running_loss = 0.0
      for i, sample in enumerate(data, 0):
        optimizer.zero_grad()
        # print(sample[0])
        # print(sample[1])
        inputs = sample[0]
        # img = np.reshape(inputs, (1, 1, 28, 28)) / 255
        # img = torch.from_numpy(img)
        # img = img.type(torch.FloatTensor)
        labels = sample[1]

        output = model(inputs)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 499:  # print every 500 mini-batches
          print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
          s = "{0} {1}\n".format(p, running_loss / 500)
          f.write(s)
          p += 1
          running_loss = 0.0

  torch.save(model.state_dict(), './model.pth')


def validation(data, model):
  model.eval()
  print("Validating...")
  show_image = False

  size = len(data)
  num_incorrect = 0
  i = 0
  for sample in data:
    images, labels = sample
    # img = transforms.functional.to_pil_image(images[0][0], mode='L')
    # img.save("img_{}.png".format(i), "png")
    output = model(images)
    predicted = torch.max(output.data, 1)
    if labels[0] != predicted[1].item():
      num_incorrect += 1
      if show_image:
        s = "Real: {0}\t Predicted: {1}".format(labels[0], predicted[1].item())
        print(s)
        my_imshow(torchvision.utils.make_grid(images))
    i += 1
  print("Validation Error: {0} %".format(100.0 * num_incorrect / size))


def sliding_window(model, image, size):
  # Here we use an RGB version of image, so it can be displayed.
  # In your expiriment, use 'numbers.png'
  # numbers_img = Image.open(image)

  stride = 2
  imageCopy = image.copy().convert('RGB')

  width = imageCopy.width
  height = imageCopy.height
  predictedObjects = []

  for x in range(0, width - size[0], stride):
    for y in range(0, height - size[1], stride):
      area = (x, y, x + size[0], y + size[1])
      roi = imageCopy.crop(area)

      transform = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(), torchvision.transforms.Resize(28), torchvision.transforms.ToTensor()])
      nim = transform(roi)
      nim = nim.view(-1, 1, size[0], size[1])

      predictedVal, predictedClass = validate(nim, model)
      if predictedVal > 1:
        predictedObjects.append([x, y, size, predictedVal, predictedClass])

  predictedObjects.sort(reverse=True, key=thirdIndex)

  removedElements = True
  while removedElements:
    removedElements = False
    print('.')
    for elem in predictedObjects:
      print(elem)
      for secondElem in reversed(predictedObjects):
        if secondElem == elem:
          continue
        if collides(Rect(elem[0], elem[1], elem[2][0], elem[2][1]), Rect(secondElem[0], secondElem[1], secondElem[2][0], secondElem[2][1])):
          predictedObjects.remove(secondElem)
          removedElements = True

  print('After removing: ')
  for elem in predictedObjects:
    print(elem)
    draw = ImageDraw.Draw(imageCopy)
    draw.rectangle([elem[0], elem[1], elem[0] + elem[2][0], elem[1] + elem[2][1]])
    draw.text([elem[0] + 2, elem[1] + 2], str(elem[4]), fill=(77, 175, 74))

  imageCopy.save('./out/detectedNumbers.png')

  imshow(np.asarray(imageCopy))
  plt.show()


def thirdIndex(element):
  return element[3]


class Rect:
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height


def collides(rect1, rect2):
  return rect1.x < rect2.x + rect2.width and rect1.x + rect1.width > rect2.x and rect1.y < rect2.y + rect2.height and rect1.y + rect1.height > rect2.y


def validate(image, model):
  out = model(image)
  predicted = torch.max(out.data, 1)
  return predicted[0].item(), predicted[1].item()


def main():
  transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.Resize(28), torchvision.transforms.ToTensor()])

  batch_size_train = 16

  train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
                                             batch_size=batch_size_train, shuffle=True)
  test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform))

  # trainfolder = datasets.ImageFolder("train", transform)
  # train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=batch_size_train, shuffle=True)

  # create instance of a model
  model = LeNet()

  # train new model
  # train(train_loader, model)

  # use existing model
  model.load_state_dict(torch.load('./model.pth'))

  # validation(test_loader, model)

  # uncoment to run sliding window
  # img = Image.open('./numbers_rgb.png')
  img = Image.open('./numbers.png')
  print(img.format)
  print(img.size)
  print(img.mode)
  # img.show()

  sliding_window(model, img, (28, 28))


if __name__ == '__main__':
  main()

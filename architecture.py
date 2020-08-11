class MyNet13(nn.Module):
    def __init__(self):
        super(MyNet13, self).__init__()
        self.name = "net"
        self.conv1 = nn.Conv1d(2, 64, 11) # input channel is 2 for audio files
        self.conv2 = nn.Conv1d(64, 64, 9)
        self.conv3 = nn.Conv1d(64, 64, 7)
        self.conv4 = nn.Conv1d(64, 32, 5)
        self.conv5 = nn.Conv1d(32, 16, 5)
        self.conv6 = nn.Conv1d(16, 2, 5)
        self.fc1 = nn.Linear(254, num_classes)
        self.pool = nn.MaxPool1d(2, 2) 
        self.pool2 = nn.MaxPool1d(4, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)
        self.bn6 = nn.BatchNorm1d(2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x)))) 
        x = self.bn2(self.pool(F.relu(self.conv2(x)))) 
        x = self.bn3(self.pool2(F.relu(self.conv3(x)))) 
        x = self.bn4(self.pool2(F.relu(self.conv4(x))))
        x = self.bn5(self.pool2(F.relu(self.conv5(x))))
        x = self.bn6(self.pool2(F.relu(self.conv6(x))))
        # print(np.shape(x))
        x = self.drop(x)
        x = x.view(-1, 254)
        x = self.fc1(x)
        return x

class TransferModel(nn.Module): # This model will output an embedding state with shape [batch_num, 514]
  def __init__(self):
    super(TransferModel, self).__init__()
    self.name = "transferNet"
    self.conv1 = nn.Conv1d(2, 64, 11) # input channel is 2 for audio files
    self.conv2 = nn.Conv1d(64, 64, 9)
    self.conv3 = nn.Conv1d(64, 64, 7)
    self.conv4 = nn.Conv1d(64, 32, 5)
    self.conv5 = nn.Conv1d(32, 16, 5)
    self.conv6 = nn.Conv1d(16, 2, 5)
    self.pool = nn.MaxPool1d(2, 2) 
    self.pool2 = nn.MaxPool1d(4, 4)
    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(64)
    self.bn4 = nn.BatchNorm1d(32)
    self.bn5 = nn.BatchNorm1d(16)
    self.bn6 = nn.BatchNorm1d(2)

  def forward(self, x):
    x = self.bn1(self.pool(F.relu(self.conv1(x)))) 
    x = self.bn2(self.pool(F.relu(self.conv2(x)))) 
    x = self.bn3(self.pool2(F.relu(self.conv3(x)))) 
    x = self.bn4(self.pool2(F.relu(self.conv4(x))))
    x = self.bn5(self.pool2(F.relu(self.conv5(x))))
    x = self.bn6(self.pool2(F.relu(self.conv6(x))))
    x = x.view(-1, 254)
    return x

def LoadFeatureModel(state_dict_path, transfered_model):
  # Load the best MyNet13 model
  MyNet13_best_state = torch.load(state_dict_path) #The state_dict file is stored in the shared google drive
  MyNet13_model = MyNet13()
  MyNet13_model.load_state_dict(MyNet13_best_state)

  # Copy features from MyNet11 to Transfered_model
  transfered_model.conv1 = MyNet13_model.conv1
  transfered_model.conv2 = MyNet13_model.conv2
  transfered_model.conv3 = MyNet13_model.conv3
  transfered_model.conv4 = MyNet13_model.conv4
  transfered_model.conv5 = MyNet13_model.conv5
  transfered_model.conv6 = MyNet13_model.conv6
  transfered_model.bn1 = MyNet13_model.bn1
  transfered_model.bn2 = MyNet13_model.bn2
  transfered_model.bn3 = MyNet13_model.bn3
  transfered_model.bn4 = MyNet13_model.bn4
  transfered_model.bn5 = MyNet13_model.bn5
  transfered_model.bn6 = MyNet13_model.bn6

  # Disable gradient for transfered_model
  for param in transfered_model.parameters():
      param.requires_grad = False

  return transfered_model

def LoadFeature(transfered_model, original_folder, batch_size=64): # Output the feature dataset
  train_loader, val_loader, test_loader = get_data_loaders(combined_audio_folder, batch_size=64)
  
  feature_train_loader = []
  feature_val_loader = []
  feature_test_loader = []

  if use_cuda and torch.cuda.is_available():
    transfered_model = transfered_model.cuda()

  for inputs, labels in train_loader:
    if use_cuda and torch.cuda.is_available():
      inputs = inputs.cuda()
      labels = labels.cuda()
    features = transfered_model(inputs)
    feature_train_loader.append([features, labels])

  for inputs, labels in val_loader:
    if use_cuda and torch.cuda.is_available():
      inputs = inputs.cuda()
      labels = labels.cuda()
    features = transfered_model(inputs)
    feature_val_loader.append([features, labels]) 

  for inputs, labels in test_loader:
    if use_cuda and torch.cuda.is_available():
      inputs = inputs.cuda()
      labels = labels.cuda()
    features = transfered_model(inputs)
    feature_test_loader.append([features, labels]) 

  # Avoid pytorch to track weight update in feature data 
  #features = torch.from_numpy(features.detach().numpy())

  return feature_train_loader, feature_val_loader, feature_test_loader

class predictionNet(nn.Module):
    def __init__(self):
        super(predictionNet, self).__init__()
        self.name = "prediction_net"
        self.fc1 = nn.Linear(254, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
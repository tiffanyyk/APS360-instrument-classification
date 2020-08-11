def get_accuracy(model, loader):
    correct = 0
    total = 0
    for inputs, labels in loader:
        if use_cuda and torch.cuda.is_available():
           inputs = inputs.cuda()
           labels = labels.cuda()
        output = model(inputs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def train(model, train_loader=None, valid_loader=None, batch_size=64, num_epochs=5, learning_rate=1e-4, checkpoint=False, checkpoint_name=None, checkpoint_bestonly=False): 
    torch.manual_seed(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_valacc = 0.0

    if train_loader is not None and valid_loader is not None:
        pass
    else:
        train_loader, val_loader, _ = get_data_loaders(audioFolder, batch_size) 

    epoch_plot, losses, val_losses, train_acc, val_acc = [], [], [], [], []
    for epoch in range(num_epochs):
        total_train_loss = 0
        num_train_batch = 0
        for inputs, labels in iter(train_loader):

            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss = total_train_loss + (float(loss.item()) /len(inputs))
            num_train_batch += 1
        total_train_loss = total_train_loss / num_train_batch #/ 21502
        losses.append(float(total_train_loss))
        train_acc.append(get_accuracy(model,train_loader))
        
        # make validation predictions and calculate loss
        total_val_loss = 0
        num_val_batch = 0
        for inputs, labels in iter(val_loader):
            
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            pred = model(inputs)
            val_loss = criterion(pred,labels)
            
            total_val_loss = total_val_loss + (float(val_loss.item()) /len(inputs))
            num_val_batch += 1
        total_val_loss = total_val_loss / num_val_batch #/ 4608
        val_losses.append(float(total_val_loss))
        val_acc.append(get_accuracy(model,val_loader))

        epoch_plot.append(epoch)
        print('Epoch:{}, Loss:{:.4f}, Val_Loss:{:.4f}, Train_acc:{:.4f}, Val_acc:{:.4f}'.format(
            epoch+1,
            float(total_train_loss),
            float(total_val_loss),
            float(train_acc[epoch]),
            float(val_acc[epoch])))

        # Save the current model (checkpoint) to a file
        if checkpoint:
            if (checkpoint_bestonly and val_acc[-1] > best_valacc):
                best_valacc = val_acc[-1]
                if checkpoint_name is not None:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_best".format(checkpoint_name,batch_size,learning_rate,epoch)
                else:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}_best".format(batch_size,learning_rate,epoch)
                torch.save(model.state_dict(), model_path)
            elif not checkpoint_bestonly:
                if checkpoint_name is not None:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_epoch={}".format(checkpoint_name,batch_size,learning_rate,epoch)
                else:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}_epoch={}".format(batch_size,learning_rate,epoch)
                torch.save(model.state_dict(), model_path)

    # plotting
    plt.title("Training Curve")
    plt.plot(epoch_plot, losses, label="Train")
    plt.plot(epoch_plot, val_losses, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(epoch_plot, train_acc, label="Train")
    plt.plot(epoch_plot, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Maximum validation accuracy for this model is:", max(val_acc),
           "at epoch", epoch_plot[val_acc.index(max(val_acc))],"\n")

def get_accuracy_transfer_learning(prediction_model, loader):
    """
    Model output is considered correct only if all four outputs are correct.
    """
    correct = 0
    total = 0

    t = torch.Tensor([0])

    for features, labels in loader:
        
        if use_cuda and torch.cuda.is_available():
          features = features.cuda()
          labels = labels.cuda()
          t = t.cuda()
        
        outputs = prediction_model(features)
        one_hot_outputs = (outputs >= t).int()

        corr = sum(sum(one_hot_outputs == labels)).item()

        correct += corr
        total += labels.shape[0] * 4
    
    return correct / total

def transfer_train(prediction_model, train_loader=None, val_loader=None, batch_size=64, num_epochs=5, 
            learning_rate=1e-4, checkpoint=False, checkpoint_name=None, checkpoint_bestonly=False,
            accuracy=get_accuracy_transfer_learning): 
    torch.manual_seed(1)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(prediction_model.parameters(), lr=learning_rate)
    best_valacc = 0.0

    if train_loader is not None and val_loader is not None:
        pass
    else:
        train_loader, val_loader, _ = get_data_loaders(combined_audio_folder, batch_size) 

    epoch_plot, losses, val_losses, train_acc, val_acc = [], [], [], [], []

    for epoch in range(num_epochs):
        total_train_loss = 0
        start_time = time.time()
        prediction_model.train()
        for i, (features, labels) in enumerate(train_loader, 0):

            if use_cuda and torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()

            outputs = prediction_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
        total_train_loss = total_train_loss / (i+1) #/ 21502
        losses.append(total_train_loss)
        train_acc.append(accuracy(prediction_model,train_loader))
        
        # make validation predictions and calculate loss
        total_val_loss = 0
        num_val_batch = 0
        prediction_model.eval()
        with torch.no_grad():
          for features, labels in iter(val_loader):

              if use_cuda and torch.cuda.is_available():
                  features = features.cuda()
                  labels = labels.cuda()

              outputs = prediction_model(features)
              
              val_loss = criterion(outputs, labels)

              total_val_loss += val_loss.item()
              num_val_batch += 1
        total_val_loss = total_val_loss / num_val_batch #/ 4608
        val_losses.append(float(total_val_loss))
        val_acc.append(accuracy(prediction_model, val_loader))

        epoch_plot.append(epoch+1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Epoch:{}, Loss:{:.4f}, Val_Loss:{:.4f}, Train_acc:{:.4f}, Val_acc:{:.4f}, Total time elapsed: {:.2f} seconds'.format(
            epoch+1,
            float(total_train_loss),
            float(total_val_loss),
            float(train_acc[epoch]),
            float(val_acc[epoch]),
            elapsed_time))

        # Save the current model (checkpoint) to a file
        if checkpoint:
            if (checkpoint_bestonly and val_acc[-1] > best_valacc):
                best_valacc = val_acc[-1]
                best_epoch = epoch+1
                best_model_state = prediction_model.state_dict()
            elif not checkpoint_bestonly:
                if checkpoint_name is not None:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_epoch={}".format(checkpoint_name,batch_size,learning_rate,epoch)
                else:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}_epoch={}".format(batch_size,learning_rate,epoch)
                torch.save(prediction_model.state_dict(), model_path)


    if checkpoint_name is not None:
        model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_epoch={}_best".format(checkpoint_name,batch_size,learning_rate,best_epoch)
    else:
        model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}__epoch={}_best".format(batch_size,learning_rate,best_epoch)
    torch.save(best_model_state, model_path)

    # plotting
    plt.title("Training Curve")
    plt.plot(epoch_plot, losses, label="Train")
    plt.plot(epoch_plot, val_losses, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(epoch_plot, train_acc, label="Train")
    plt.plot(epoch_plot, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Maximum validation accuracy for this model is:", max(val_acc),
           "at epoch", epoch_plot[val_acc.index(max(val_acc))],"\n")

def get_accuracy_multilabel(model, loader):
    """
    Model output is considered correct only if all four outputs are correct.
    """
    correct = 0
    total = 0
    for inputs, labels in loader:
        if use_cuda and torch.cuda.is_available():
           inputs = inputs.cuda()
           labels = labels.cuda()
        outputs = model(inputs)
        zeros = torch.from_numpy(np.zeros(np.shape(outputs))).cuda() if (use_cuda and torch.cuda.is_available()) else torch.from_numpy(np.zeros(np.shape(outputs)))
        batch_size = inputs.shape[0]
        corr = [True if all((outputs[i,:]>zeros[i,:]).long()==labels[i,:]) else False for i in range(batch_size)]
        # print(corr)
        correct += int(sum(corr))
        total += inputs.shape[0]
    return correct / total

def get_part_accuracy_multilabel(model,loader):
    """
    "Part marks" assigned for calculating model output correctness. Each correct
    binary classification is considered, even if other outputs corresponding to 
    the same data sample are incorrect.
    """
    correct = 0
    total = 0
    t = torch.Tensor([0])
    for inputs, labels in loader:   
        if use_cuda and torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()
          t = t.cuda()
        outputs = model(inputs)
        one_hot_outputs = (outputs >= t).int()
        corr = sum(sum(one_hot_outputs == labels)).item()
        correct += corr
        total += inputs.shape[0] * 4
    return correct / total


def train_multilabel(model, train_loader=None, valid_loader=None, batch_size=64, num_epochs=5, 
            learning_rate=1e-4, checkpoint=False, checkpoint_name=None, checkpoint_bestonly=False): 
    torch.manual_seed(1)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_valacc = 0.0

    if train_loader is not None and valid_loader is not None:
        pass
    else:
        train_loader, val_loader, _ = get_data_loaders(combined_audio_folder, batch_size) 

    epoch_plot, losses, val_losses, train_acc, train_acc_part, val_acc, val_acc_part = [], [], [], [], [], [], []
    for epoch in range(num_epochs):
        total_train_loss = 0
        num_train_batch = 0
        for inputs, labels in iter(train_loader):

            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss = total_train_loss + (float(loss.item()) /len(inputs))
            num_train_batch += 1
        total_train_loss = total_train_loss / num_train_batch #/ 21502
        losses.append(float(total_train_loss))
        train_acc.append(get_accuracy_multilabel(model,train_loader))
        train_acc_part.append(get_part_accuracy_multilabel(model,train_loader))
        
        # make validation predictions and calculate loss
        total_val_loss = 0
        num_val_batch = 0
        for inputs, labels in iter(val_loader):
            
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            pred = model(inputs)
            val_loss = criterion(pred,labels)
            
            total_val_loss = total_val_loss + (float(val_loss.item()) /len(inputs))
            num_val_batch += 1
        total_val_loss = total_val_loss / num_val_batch #/ 4608
        val_losses.append(float(total_val_loss))
        val_acc.append(get_accuracy_multilabel(model,val_loader))
        val_acc_part.append(get_part_accuracy_multilabel(model,val_loader))

        epoch_plot.append(epoch)
        print('Epoch:{}, Loss:{:.4f}, Val_Loss:{:.4f}, Train_acc:{:.4f}, Train_acc_part:{:.4f}, Val_acc:{:.4f}, Val_acc_part:{:.4f}'.format(
            epoch+1,
            float(total_train_loss),
            float(total_val_loss),
            float(train_acc[epoch]),
            float(train_acc_part[epoch]),
            float(val_acc[epoch]),
            float(val_acc_part[epoch])))

        # Save the current model (checkpoint) to a file
        if checkpoint:
            if (checkpoint_bestonly and val_acc_part[-1] > best_valacc):
                best_valacc = val_acc_part[-1]
                if checkpoint_name is not None:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_best".format(checkpoint_name,batch_size,learning_rate,epoch)
                else:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}_best".format(batch_size,learning_rate,epoch)
                torch.save(model.state_dict(), model_path)
            elif not checkpoint_bestonly:
                if checkpoint_name is not None:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/{}_batch_size={}_lr={}_epoch={}".format(checkpoint_name,batch_size,learning_rate,epoch)
                else:
                    model_path = "/content/drive/My Drive/APS 360 Project/saved_models/batch_size={}_lr={}_epoch={}".format(batch_size,learning_rate,epoch)
                torch.save(model.state_dict(), model_path)

    # plotting
    plt.title("Training Curve")
    plt.plot(epoch_plot, losses, label="Train")
    plt.plot(epoch_plot, val_losses, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(epoch_plot, train_acc, label="Train")
    plt.plot(epoch_plot, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(epoch_plot, train_acc_part, label="Train")
    plt.plot(epoch_plot, val_acc_part, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Part Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Training (part) Accuracy: {}".format(train_acc_part[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print("Final Validation (part) Accuracy: {}".format(val_acc_part[-1]))
    print ("Maximum validation accuracy for this model is:", max(val_acc),
           "at epoch", epoch_plot[val_acc.index(max(val_acc))],"\n")
    print ("Maximum validation (part) accuracy for this model is:", max(val_acc_part),
           "at epoch", epoch_plot[val_acc_part.index(max(val_acc_part))],"\n")
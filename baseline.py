import preprocessing


def accuracy_int_bit(value1,value2):
      accuracy = 0
  for i in range(3):
    if f'{value1:04b}'[i] == f'{value2:04b}'[i]:
      accuracy += 1
  return accuracy/4


if __name__ == '__main__':
    # Stage 1:

    train_inputs = []
    train_labels = []
    for i,data in enumerate(train_loader,0):
        inputs,labels = data
        train_inputs.append(torch.reshape(inputs[0],(-1,)).numpy())
        train_labels.append(labels.item())
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    val_inputs = []
    val_labels = []
    for i,data in enumerate(val_loader,0):
        inputs,labels = data
        val_inputs.append(torch.reshape(inputs[0],(-1,)).numpy())
        val_labels.append(labels.item())

    val_inputs = np.array(val_inputs)
    val_labels = np.array(val_labels)

    #baseline model training here
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=500)

    # Fit the model to our training data
    model.fit(train_inputs, train_labels)

    # Make predictions
    val_predicted = model.predict(val_inputs)

    correct = 0
    for i in range(len(val_predicted)):
        if val_predicted[i] == val_labels[i]:
            correct +=1 
    print("accuracy of baseline model: {0}".format(correct/len(val_predicted)))

    # *********************************************************************************************
    # Stage 2:

    train_loader, val_loader, test_loader = get_data_loaders(combined_audio_folder, 1)

    # Output the size of each dataset.
    print("# of training examples: ", len(train_loader))
    print("# of validation examples: ", len(val_loader))
    print("# of test examples: ", len(test_loader))

    train_inputs = []
    train_labels = []
    for i,data in enumerate(train_loader,0):
        inputs,labels = data
        train_inputs.append(torch.reshape(inputs[0],(-1,)).numpy())
        res = int("".join(str(x) for x in np.array(torch.reshape(labels[0],(-1,)).numpy())), 2)  
        train_labels.append(res)
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    val_inputs = []
    val_labels = []
    for i,data in enumerate(val_loader,0):
        inputs,labels = data
        val_inputs.append(torch.reshape(inputs[0],(-1,)).numpy())
        res = int("".join(str(x) for x in np.array(torch.reshape(labels[0],(-1,)).numpy())), 2)  
        val_labels.append(res)

    val_inputs = np.array(val_inputs)
    al_labels = np.array(val_labels)

    #baseline model training here
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)

    # Fit the model to our training data
    model.fit(train_inputs, train_labels)

    # Make predictions
    val_predicted = model.predict(val_inputs)

    correct = 0
    partial_correct =  0
    for i in range(len(val_predicted)):
        if val_predicted[i] == val_labels[i]:
            correct +=1 
        partial_correct += accuracy_int_bit(val_predicted[i],val_labels[i])
    print("accuracy of baseline model: {0}".format(correct/len(val_predicted)))
    print("Partial accuracy of baseline model: {0}".format(partial_correct/len(val_predicted)))
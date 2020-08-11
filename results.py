import train
import architecture
import preprocessing
from sklearn.metrics import confusion_matrix

train_loader1, val_loader1, test_loader1 = get_data_loaders(audioFolder, 16) 
train_loader2, val_loader2, test_loader2 = get_data_loaders(combined_audio_folder, 16) 

def multi_hot_to_num(label):
    # print(label)
    if label == [1,1,0,0]:
        return 0
    elif label == [1,0,1,0]:
        return 1
    elif label == [1,0,0,1]:
        return 2
    elif label == [0,1,1,0]:
        return 3
    elif label == [0,1,0,1]:
        return 4
    elif label == [0,0,1,1]:
        return 5
    else:
        # print("Error")
        return 6

if __name__ == '__main__':
    # load stage 1 model
    # load in best model and check training accuracy
    stage_1_model = MyNet13()
    saved_model = '/saved_models/MyNet13_batch_size=16_lr=0.0003_best_0.7257valacc'
    stage_1_model.load_state_dict(torch.load(saved_model,map_location=torch.device('cpu')))

    # # do test set predictions for stage 1
    print("Overall Test Accuracy (Stage 1):",get_accuracy(stage_1_model.eval().cuda(),test_loader1))
    print("Confusion Matrix:")
    stage_1_model = stage_1_model.eval().cuda()
    all_outputs = []
    all_labels = []
    for inputs, labels in test_loader1:
        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = stage_1_model(inputs)
        pred = outputs.max(1, keepdim=True)[1].view_as(labels)
        all_outputs.extend(pred.tolist())
        all_labels.extend(labels.tolist())
    print(confusion_matrix(all_labels,all_outputs))
    
    stage_2_model = MyNet13()
    saved_model = '/saved_models/Stage2_MyNet13_batch_size=16_lr=0.0003_best_0.6905valaccpart_0.2243valacc'
    stage_2_model.load_state_dict(torch.load(saved_model,map_location=torch.device('cpu')))

    # do test set predictions for stage 2 (non transfer learning)
    # print(get_part_accuracy_multilabel_class(stage_2_model.eval().cuda(),test_loader2,0))
    print("Overall Test Accuracy (Stage 2)", get_part_accuracy_multilabel(stage_2_model.eval().cuda(),test_loader2))
    stage_2_model = stage_2_model.eval().cuda()
    all_outputs = []
    all_labels = []
    t = torch.Tensor([0]).cuda()
    for inputs, labels in test_loader2:
        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = stage_2_model(inputs)
        pred = (outputs >= t).int()
        all_outputs.extend(pred.tolist())
        all_labels.extend(labels.tolist())

    converted_labels = [multi_hot_to_num(label) for label in all_labels]
    converted_outputs = [multi_hot_to_num(output) for output in all_outputs]
    print(confusion_matrix(converted_labels,converted_outputs))
        
import preprocessing
import architecture
import train



if __name__ == '__main__':
    # Stage 1 training:


    torch.cuda.empty_cache()
    model = MyNet8()
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    train(model, learning_rate=0.0003, num_epochs=30, batch_size=32, checkpoint=True, checkpoint_name='MyNet8', checkpoint_bestonly=True)

    # Testing
    stage_1_model = MyNet13()
    saved_model = '/saved_models/MyNet13_batch_size=16_lr=0.0003_best_0.7257valacc'
    stage_1_model.load_state_dict(torch.load(saved_model))
    train_loader, val_loader, test_loader = get_data_loaders(audioFolder, 16) 
    print(get_accuracy(stage_1_model.eval().cuda(),test_loader))

    # Stage 2 training:

    torch.cuda.empty_cache()
    model2 = MyNet13()
    if use_cuda and torch.cuda.is_available():
        model2 = model2.cuda()
    train_multilabel(model2, learning_rate=0.0003, num_epochs=100, batch_size=16, checkpoint=True, checkpoint_name='Stage2_MyNet13', checkpoint_bestonly=True)


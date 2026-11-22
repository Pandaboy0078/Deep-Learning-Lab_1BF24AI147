from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from lab01 import fashion_experiments as exp

def test_experiment_pipeline_creates_outputs(tmp_path,monkeypatch):
    g=torch.Generator().manual_seed(42)
    train_x=torch.randn(40,1,28,28,generator=g)
    train_y=torch.arange(40)%10
    val_x=torch.randn(20,1,28,28,generator=g)
    val_y=torch.arange(20)%10
    def loader(x,y,shuffle=False):
        return DataLoader(TensorDataset(x,y),batch_size=20,shuffle=shuffle,
                          generator=torch.Generator().manual_seed(42))
    def fake_loaders(*args,**kwargs):
        return dict(train=loader(train_x,train_y,True),
                    clean_train=loader(train_x,train_y),
                    augmented_train=loader(train_x,train_y,True),
                    small_train=loader(train_x,train_y),
                    small_shuffled_train=loader(train_x,train_y,True),
                    small_augmented_train=loader(train_x,train_y,True),
                    validation=loader(val_x,val_y),
                    test=loader(val_x,val_y))
    monkeypatch.setattr(exp,"build_loaders",fake_loaders)
    rows,selected=exp.run_experiments(tmp_path,tmp_path,mode="quick",device_name="cpu")
    assert len(rows)==18
    assert len(selected)==3
    for file in ("validation_results.csv","selected_test_results.csv",
                 "epoch_histories.csv","optimizer_validation_accuracy.png",
                 "learning_rate_schedules.png","overfitting_train_vs_val_loss.png",
                 "regularization_validation_loss.png","architecture_validation_accuracy.png",
                 "confusion_matrix_optimizers.png","confusion_matrix_regularization.png",
                 "confusion_matrix_architecture.png"):
        assert (Path(tmp_path)/file).exists()

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from lab01.fashion_experiments import (MLP,OPTIMIZERS,balanced_subsample,
    evaluate,make_optimizer,seed_everything,stratified_indices,train_model)

def test_balanced_disjoint_split():
    targets=np.repeat(np.arange(10),30)
    train,val=stratified_indices(targets,100,50,42)
    assert len(train)==100 and len(val)==50
    assert set(train).isdisjoint(val)
    assert np.bincount(targets[train]).tolist()==[10]*10
    assert np.bincount(targets[val]).tolist()==[5]*10
    small=balanced_subsample(train,targets,20,42)
    assert np.bincount(targets[small]).tolist()==[2]*10

def test_same_parameter_count():
    seed_everything(42)
    plain=MLP(width=32)
    seed_everything(42)
    residual=MLP(width=32,residual=True)
    assert sum(p.numel() for p in plain.parameters())==sum(p.numel() for p in residual.parameters())
    assert plain(torch.randn(8,1,28,28)).shape==(8,10)
    assert residual(torch.randn(8,1,28,28)).shape==(8,10)

def test_one_epoch_training():
    seed_everything(7)
    x=torch.randn(40,1,28,28);y=torch.arange(40)%10
    loader=DataLoader(TensorDataset(x,y),batch_size=10)
    model=MLP(width=32)
    history,stats,state=train_model(model,loader,loader,loader,
                                     torch.device("cpu"),"Adam",.001,1)
    assert len(history["val_loss"])==1 and state
    assert 0<=stats["best_val_accuracy"]<=1
    assert 0<=evaluate(model,loader,torch.device("cpu"))[1]<=1

def test_optimizer_updates():
    for name,lr in OPTIMIZERS.items():
        model=MLP(width=16)
        optimizer=make_optimizer(name,model,lr)
        loss=torch.nn.functional.cross_entropy(model(torch.randn(4,1,28,28)),
                                                 torch.tensor([0,1,2,3]))
        loss.backward();optimizer.step()
        assert torch.isfinite(loss)

def test_scheduled_lr():
    x=torch.randn(20,1,28,28);y=torch.arange(20)%10
    loader=DataLoader(TensorDataset(x,y),batch_size=10)
    history,stats,_=train_model(MLP(width=16),loader,loader,loader,
                                 torch.device("cpu"),"Adam",.001,3,scheduled=True)
    assert history["learning_rate"][0]==.001
    assert history["learning_rate"][1]<history["learning_rate"][0]
    assert stats["epochs_ran"]==3

"""Fashion-MNIST experiments: optimizers, regularization, and residual MLPs."""
import copy
import csv
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

OPTIMIZERS={"SGD_momentum":0.025,"Nesterov":0.025,"RMSprop":0.001,"Adam":0.001,"AdamW":0.001}
CLASS_NAMES=("T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal",
             "Shirt","Sneaker","Bag","Ankle boot")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_device(requested="auto"):
    if requested!="auto": return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def stratified_indices(targets,train_count,val_count,seed):
    targets=np.asarray(targets)
    classes=np.unique(targets)
    if len(classes)==0 or train_count<=0 or val_count<0 or train_count%len(classes) or val_count%len(classes):
        raise ValueError("Sample counts must be positive and class-balanced")
    rng=np.random.default_rng(seed)
    train=[]; val=[]
    for label in classes:
        candidates=rng.permutation(np.flatnonzero(targets==label))
        a=train_count//len(classes); b=val_count//len(classes)
        if len(candidates)<a+b: raise ValueError("Insufficient class examples")
        train.extend(candidates[:a]);val.extend(candidates[a:a+b])
    rng.shuffle(train);rng.shuffle(val)
    return train,val

def balanced_subsample(indices,targets,count,seed):
    targets=np.asarray(targets);classes=np.unique(targets)
    if count<=0 or count%len(classes): raise ValueError("Invalid subset size")
    rng=np.random.default_rng(seed);ids=[]
    for label in classes:
        candidates=[i for i in indices if targets[i]==label]
        if len(candidates)<count//len(classes): raise ValueError("Insufficient examples")
        ids.extend(rng.choice(candidates,count//len(classes),replace=False).tolist())
    rng.shuffle(ids)
    return ids

class MLP(nn.Module):
    def __init__(self,width=128,dropout=0.,residual=False):
        super().__init__()
        self.residual=residual
        self.input_layer=nn.Linear(784,width)
        self.hidden_layers=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        self.dropout=nn.Dropout(dropout)
        self.output_layer=nn.Linear(width,10)
    def forward(self,x):
        x=self.dropout(torch.relu(self.input_layer(x.reshape(x.shape[0],-1))))
        for layer in self.hidden_layers:
            update=layer(x)
            x=self.dropout(torch.relu(x+update if self.residual else update))
        return self.output_layer(x)

def make_optimizer(name,model,lr,weight_decay=0.):
    params=model.parameters()
    if name=="SGD_momentum": return torch.optim.SGD(params,lr=lr,momentum=.9,weight_decay=weight_decay)
    if name=="Nesterov": return torch.optim.SGD(params,lr=lr,momentum=.9,nesterov=True,weight_decay=weight_decay)
    if name=="RMSprop": return torch.optim.RMSprop(params,lr=lr,weight_decay=weight_decay)
    if name=="Adam": return torch.optim.Adam(params,lr=lr,weight_decay=weight_decay)
    if name=="AdamW": return torch.optim.AdamW(params,lr=lr,weight_decay=weight_decay)
    raise ValueError("Unknown optimizer "+name)

def build_loaders(data_dir,train_count,val_count,test_count,seed,batch_size):
    standard=transforms.ToTensor()
    augmented=transforms.Compose([
        transforms.RandomAffine(degrees=10,translate=(.07,.07)),
        transforms.ToTensor()])
    train=datasets.FashionMNIST(data_dir,train=True,download=True,transform=standard)
    aug=datasets.FashionMNIST(data_dir,train=True,download=False,transform=augmented)
    test=datasets.FashionMNIST(data_dir,train=False,download=True,transform=standard)
    train_ids,val_ids=stratified_indices(train.targets,train_count,val_count,seed)
    test_ids,_=stratified_indices(test.targets,test_count,0,seed+1)
    small=balanced_subsample(train_ids,train.targets,min(400,train_count),seed+2)
    def loader(ds,shuffle=False):
        return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    return dict(train=loader(Subset(train,train_ids),True),
                clean_train=loader(Subset(train,train_ids)),
                augmented_train=loader(Subset(aug,train_ids),True),
                validation=loader(Subset(train,val_ids)),
                test=loader(Subset(test,test_ids)),
                small_train=loader(Subset(train,small)),
                small_shuffled_train=loader(Subset(train,small),True),
                small_augmented_train=loader(Subset(aug,small),True))

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();loss_sum=0.;correct=0;count=0
    criterion=nn.CrossEntropyLoss(reduction="sum")
    for x,y in loader:
        x=x.to(device);y=y.to(device)
        out=model(x);loss_sum+=criterion(out,y).item()
        correct+=(out.argmax(1)==y).sum().item();count+=len(y)
    return loss_sum/count,correct/count

@torch.no_grad()
def collect_predictions(model,loader,device):
    model.eval();actual=[];predicted=[]
    for x,y in loader:
        actual.extend(y.tolist())
        predicted.extend(model(x.to(device)).argmax(1).cpu().tolist())
    return np.asarray(actual),np.asarray(predicted)

def train_model(model,train_loader,clean_train_loader,val_loader,device,
                optimizer_name,learning_rate,epochs,scheduled=False,
                weight_decay=0.,early_stopping=False,patience=4):
    optimizer=make_optimizer(optimizer_name,model,learning_rate,weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,epochs//3),
                                               gamma=.5) if scheduled else None
    criterion=nn.CrossEntropyLoss()
    history={k:[] for k in ("train_loss","train_accuracy","val_loss",
                             "val_accuracy","learning_rate")}
    best_loss=float("inf");best_epoch=0;best_state=None;stalled=0
    for epoch in range(1,epochs+1):
        model.train();lr=optimizer.param_groups[0]["lr"]
        for x,y in train_loader:
            x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y);loss.backward();optimizer.step()
        train_loss,train_acc=evaluate(model,clean_train_loader,device)
        val_loss,val_acc=evaluate(model,val_loader,device)
        for key,value in (("train_loss",train_loss),("train_accuracy",train_acc),
                          ("val_loss",val_loss),("val_accuracy",val_acc),
                          ("learning_rate",lr)):
            history[key].append(float(value))
        if val_loss<best_loss-1e-4:
            best_loss=val_loss;best_epoch=epoch;stalled=0
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else: stalled+=1
        if scheduler: scheduler.step()
        if early_stopping and stalled>=patience: break
    model.load_state_dict(best_state)
    i=best_epoch-1
    stats=dict(epochs_ran=len(history["val_loss"]),best_epoch=best_epoch,
               best_val_loss=best_loss,best_val_accuracy=history["val_accuracy"][i],
               train_accuracy_at_best=history["train_accuracy"][i],
               generalization_gap=history["train_accuracy"][i]-history["val_accuracy"][i])
    return history,stats,best_state

def plot_curves(histories,metric,path,title):
    fig,ax=plt.subplots(figsize=(10,5))
    for label,h in histories.items():
        ax.plot(range(1,len(h[metric])+1),h[metric],label=label)
    ax.set(title=title,xlabel="Epoch",ylabel=metric.replace("_"," "))
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def save_confusion_and_class_metrics(actual,predicted,output_dir,group):
    matrix=np.zeros((10,10),dtype=int)
    np.add.at(matrix,(actual,predicted),1)
    with (output_dir/f"confusion_matrix_{group}.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["true/predicted",*CLASS_NAMES])
        for name,row in zip(CLASS_NAMES,matrix): w.writerow([name,*row.tolist()])
    with (output_dir/f"class_metrics_{group}.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["class","precision","recall","f1","support"])
        for i,name in enumerate(CLASS_NAMES):
            tp=matrix[i,i];support=matrix[i,:].sum();pred=matrix[:,i].sum()
            precision=tp/pred if pred else 0.;recall=tp/support if support else 0.
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.
            w.writerow([name,precision,recall,f1,support])
    fig,ax=plt.subplots(figsize=(9,7))
    image=ax.imshow(matrix,cmap="Blues");fig.colorbar(image,ax=ax)
    ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES,xlabel="Predicted",ylabel="Actual",
           title=f"Test confusion matrix: {group}")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    fig.tight_layout();fig.savefig(output_dir/f"confusion_matrix_{group}.png",dpi=160);plt.close(fig)

def run_experiments(output_dir,data_dir,mode="quick",device_name="auto",seed=42,
                    groups=("optimizers","regularization","architecture")):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    seed_everything(seed);device=pick_device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    if mode=="quick":
        sizes=(1000,300,500);optimizer_epochs,reg_epochs,arch_epochs=3,7,4
    elif mode=="full":
        sizes=(6000,1500,2000);optimizer_epochs,reg_epochs,arch_epochs=12,30,15
    else: raise ValueError("Invalid mode")
    loaders=build_loaders(data_dir,*sizes,seed,64)
    rows=[];selected=[];epoch_rows=[]
    def execute(group,label,optimizer,lr,epochs,train_key="train",
                eval_train_key="clean_train",dropout=0.,weight_decay=0.,
                scheduled=False,early_stopping=False,residual=False):
        seed_everything(seed)
        for loader in loaders.values():
            if loader.generator is not None:loader.generator.manual_seed(seed)
        model=MLP(dropout=dropout,residual=residual).to(device)
        history,stats,state=train_model(model,loaders[train_key],
                                        loaders[eval_train_key],loaders["validation"],
                                        device,optimizer,lr,epochs,scheduled,
                                        weight_decay,early_stopping)
        row=dict(group=group,configuration=label,optimizer=optimizer,
                 initial_lr=lr,scheduled=scheduled,dropout=dropout,
                 weight_decay=weight_decay,augmentation="augmented" in train_key,
                 early_stopping=early_stopping,residual=residual,
                 parameters=sum(p.numel() for p in model.parameters()),**stats)
        rows.append(row)
        for i in range(len(history["val_loss"])):
            epoch_rows.append(dict(group=group,configuration=label,epoch=i+1,
                                   **{k:v[i] for k,v in history.items()}))
        print(f"{group}: {label} | best validation accuracy {stats['best_val_accuracy']:.4f}")
        return history,(row,state)
    def test_winner(group,candidates):
        row,state=min(candidates,key=lambda x:x[0]["best_val_loss"])
        model=MLP(dropout=row["dropout"],residual=row["residual"]).to(device)
        model.load_state_dict(state)
        loss,accuracy=evaluate(model,loaders["test"],device)
        actual,predicted=collect_predictions(model,loaders["test"],device)
        save_confusion_and_class_metrics(actual,predicted,output_dir,group)
        selected.append(dict(group=group,selected_configuration=row["configuration"],
                             selection_metric="minimum validation loss",
                             test_loss=loss,test_accuracy=accuracy))
        print(f"Test: {row['configuration']}, accuracy={accuracy:.4f}")
    if "optimizers" in groups:
        histories={};candidates=[]
        for optimizer,lr in OPTIMIZERS.items():
            for scheduled in (False,True):
                label=f"{optimizer} / {'scheduled' if scheduled else 'constant'}"
                h,c=execute("optimizers",label,optimizer,lr,optimizer_epochs,scheduled=scheduled)
                histories[label]=h;candidates.append(c)
        for metric,name in (("val_accuracy","optimizer_validation_accuracy.png"),
                            ("val_loss","optimizer_validation_loss.png"),
                            ("learning_rate","learning_rate_schedules.png")):
            plot_curves(histories,metric,output_dir/name,"Optimizer comparison")
        test_winner("optimizers",candidates)
    if "regularization" in groups:
        trials=[("No regularization",{}),
                ("Dropout 0.35",dict(dropout=.35)),
                ("L2 weight decay 1e-3",dict(weight_decay=.001)),
                ("Data augmentation",dict(train_key="small_augmented_train")),
                ("Early stopping",dict(early_stopping=True)),
                ("Combined",dict(dropout=.3,weight_decay=.001,
                                  train_key="small_augmented_train",early_stopping=True))]
        histories={};candidates=[]
        for label,settings in trials:
            args=dict(train_key="small_shuffled_train",eval_train_key="small_train")
            args.update(settings)
            h,c=execute("regularization",label,"Adam",.001,reg_epochs,**args)
            histories[label]=h;candidates.append(c)
        plot_curves(histories,"val_loss",output_dir/"regularization_validation_loss.png",
                    "Regularization validation loss")
        plot_curves(histories,"val_accuracy",output_dir/"regularization_validation_accuracy.png",
                    "Regularization validation accuracy")
        base=histories["No regularization"]
        plot_curves({"Training":{"val_loss":base["train_loss"]},"Validation":base},
                    "val_loss",output_dir/"overfitting_train_vs_val_loss.png",
                    "Overfitting: training and validation")
        test_winner("regularization",candidates)
    if "architecture" in groups:
        histories={};candidates=[]
        for label,residual in (("Plain MLP",False),("Residual MLP",True)):
            h,c=execute("architecture",label,"Adam",.001,arch_epochs,residual=residual)
            histories[label]=h;candidates.append(c)
        plot_curves(histories,"val_accuracy",output_dir/"architecture_validation_accuracy.png",
                    "Plain versus residual MLP")
        plot_curves(histories,"val_loss",output_dir/"architecture_validation_loss.png",
                    "Architecture validation loss")
        test_winner("architecture",candidates)
    def write_csv(name,items):
        if items:
            with (output_dir/name).open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(items[0].keys()))
                w.writeheader();w.writerows(items)
    write_csv("validation_results.csv",rows)
    write_csv("epoch_histories.csv",epoch_rows)
    write_csv("selected_test_results.csv",selected)
    return rows,selected

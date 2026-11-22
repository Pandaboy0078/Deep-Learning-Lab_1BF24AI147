"""Run individual experiments or the complete lab."""
import argparse
from pathlib import Path
from lab01.gradient_descent import run_demo
from lab01.fashion_experiments import run_experiments

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["quick","full"],default="quick")
    parser.add_argument("--group",choices=["all","gd","optimizers","regularization","architecture"],default="all")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output-dir",default="outputs")
    parser.add_argument("--data-dir",default="data")
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if args.group in ("all","gd"):
        run_demo(output,seed=args.seed)
    if args.group!="gd":
        groups=("optimizers","regularization","architecture") if args.group=="all" else (args.group,)
        run_experiments(output,args.data_dir,mode=args.mode,device_name=args.device,
                        seed=args.seed,groups=groups)
if __name__=="__main__":
    main()

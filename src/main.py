
import os
import pprint
import warnings
import neptune
from config import getConfig
from tools.utils import setup_seed
from trainer import Trainer
warnings.filterwarnings('ignore')
args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    setup_seed(seed)
    
    if args.logging:
        api = "api_token"
        run = neptune.init_run(project="ID/Project", api_token=api, name=args.experiment, tags=args.tag)
        run.assign({'parameters':vars(args)})
        exp_num = run._sys_id.split('-')[-1].zfill(3)
        
    else:
        run = None
        exp_num = args.exp_num
    
    save_path = os.path.join(args.model_path, exp_num)
    
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, run, save_path)
    
if __name__ == '__main__':
    main(args)

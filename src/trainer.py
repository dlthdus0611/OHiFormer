import torch
import torch.nn as nn 
import pandas as pd
import os
import time
from tools.utils import *
from tools.beam import *
from dataloader import ScreenCaptionDataset, collate_fn
from torch.utils.data import DataLoader
from model.network import OHiFormer
from tqdm import tqdm 
import logging
import warnings
warnings.filterwarnings('ignore')
    
class Trainer():
    def __init__(self, args, run, save_path):
        super(Trainer, self).__init__()
        self.run = run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path
        dirname = os.getcwd().split('/')[-1]
        
        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(dirname)
        self.logger.info(args.tag)

        # Train, Valid Set load
        df_train = pd.read_csv('../data/df_train.csv')
        df_val = pd.read_csv('../data/df_dev.csv')
        df_test = pd.read_csv('../data/df_test.csv')
            
        word2idx = load_json_file('../data/screen2words_vocab.json')
        self.idx2word = {v:k for k,v in word2idx.items()}
        self.sos_ind, self.eos_ind = 3, 1
        
        tr_ds = ScreenCaptionDataset(df_train, word2idx)
        vl_ds = ScreenCaptionDataset(df_val, word2idx)
        ts_ds = ScreenCaptionDataset(df_test,word2idx)

        self.logger.info(f'Len of training data: {len(tr_ds)}')
        self.logger.info(f'Len of validation data: {len(vl_ds)}')

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_id)
        
        g = torch.Generator()
        g.manual_seed(args.seed)
        
        # TrainLoader
        self.train_loader = DataLoader(tr_ds, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(vl_ds, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        self.test_loader = DataLoader(ts_ds, collate_fn=collate_fn, shuffle=False, batch_size=8)

        # Network
        self.model = OHiFormer(args).to(self.device)
        
        if args.logging:
            self.run['model params'].append( sum([i.numel() for i in self.model.parameters()]))
            
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=False)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=False) 
                    
        iter_per_epoch = len(self.train_loader)
        
        load_epoch=0
        if args.re_training_exp is not None:
            pth_files = torch.load(f'./results/{args.re_training_exp}/best_model.pth')
            load_epoch = pth_files['epoch']
            self.model.load_state_dict(pth_files['state_dict'])
            self.optimizer.load_state_dict(pth_files['optimizer'])

            sch_dict = pth_files['scheduler']
            sch_dict['total_steps'] = sch_dict['total_steps'] + args.epochs * iter_per_epoch
            self.scheduler.load_state_dict(sch_dict)

            print(f'Start {load_epoch+1} Epoch Re-training')
            for i in range(args.warm_epoch+1, load_epoch+1):
                self.scheduler.step()

        # Train / Validate
        best_bleu = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(load_epoch+1, args.epochs+1):
            self.epoch = epoch
            self.scheduler.step()
            
            # Training
            train_loss = self.training(args)
            
            # Validation
            val_bleu = self.validate(args)

            if args.logging == True:
                self.run['Train loss'].append(train_loss)
                
            # Save models
            if val_bleu > best_bleu:
                early_stopping = 0
                best_epoch = epoch
                best_bleu = val_bleu

                torch.save({"model": self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            "epoch": epoch,
                    }, os.path.join(save_path, f'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1
                
            # Early Stopping
            if early_stopping == args.patience:
                break
        
        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val BLEU 4:{val_bleu:.2f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')
        self.best_checkpoint = os.path.join(self.save_path, 'best_model.pth')
        self.test(args)
        
    # Training
    def training(self, args):
        self.model.train()
        
        batch_losses = AverageMeter()

        for _, train_batch in tqdm(enumerate(self.train_loader),total=len(self.train_loader)):
            train_batch = {k:(v.to(self.device) if (k != 'references') and (k!='node_id') else v) for k,v in train_batch.items() } 
            targets = train_batch['screen_caption_token_ids'].squeeze(1)
            summary = torch.nn.functional.pad(targets, (1,0,0,0), value=self.sos_ind)
            summary = summary[:, :-1]

            self.optimizer.zero_grad()
            
            logits, attn_probs, dec_enc_attn_probs  = self.model(train_batch, summary)
            loss = self.criterion(logits.permute(0,2,1), targets.long()) 

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_norm)
            self.optimizer.step()
            batch_losses.update(loss.cpu().item())

        train_loss = batch_losses.avg
        current_lr = [param_group['lr'] for param_group in self.optimizer.param_groups][0]

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss:.3f} | LR:{current_lr:2.2e}')
        
        return train_loss
            
    # Validation or Dev
    def validate(self, args):
        batch_losses = AverageMeter()
        self.model.eval()
        
        with torch.no_grad():
            hyp_all = [] # predict summary
            ref_all = [] # true summary

            for _, eval_batch in (enumerate(self.val_loader)):
                eval_batch = {k:(v.to(self.device) if (k != 'references') and (k!='node_id') else v) for k,v in eval_batch.items() } 
                output = beam_decode(eval_batch, self.model, self.sos_ind, self.eos_ind, self.device, beam_width=args.beam_size)
                output = output[:, 1:].int()
                hyp_batch = torch.zeros(output.shape).fill_(self.eos_ind)
                ref_cap = eval_batch['references']
                
                for idx in range(len(ref_cap)):
                    ref_lst = ref_cap[idx].decode().split('|')
                    ref_lst = [r.strip().split(' ') for r in ref_lst if r.strip()]
                    ref_all.append(ref_lst)
                    
                for i in range(output.shape[0]):  # batch_size
                    for j in range(output.shape[1]):
                        hyp_batch[i, j] = output[i, j]
                        if output[i, j] == self.eos_ind:
                            break
                        elif j == output.shape[1] - 1:
                            hyp_batch[i, j] = self.eos_ind 
                hyp_batch = hyp_batch.int()
                hyp_all.extend(hyp_batch)
                
            metrics = decode_output(hyp_all, ref_all, self.idx2word)
            
            if args.logging == True:
                self.run['val BLEU-1'].append(metrics['BLEU-1'])
                self.run['val BLEU-2'].append(metrics['BLEU-2'])
                self.run['val BLEU-3'].append(metrics['BLEU-3'])
                self.run['val BLEU-4'].append(metrics['BLEU-4'])
                self.run['val ROUGE-L'].append(metrics['ROUGE-L'])
                self.run['val CIDEr'].append(metrics['CIDEr'])

            return metrics["BLEU-4"]

    def test(self, args):
        weights = torch.load((self.best_checkpoint))['model']
        
        self.model.load_state_dict(weights)
        self.model.eval()
        
        with torch.no_grad():
            hyp_all = []
            ref_all = []
            
            for _, eval_batch in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                eval_batch = {k:(v.to(self.device) if (k != 'references') and (k!='node_id') else v) for k,v in eval_batch.items() } 
                output = beam_decode(eval_batch, self.model, self.sos_ind, self.eos_ind, self.device, beam_width=args.beam_size)
                output = output[:, 1:].int()
                hyp_batch = torch.zeros(output.shape).fill_(self.eos_ind)

                ref_cap = eval_batch['references']
                for idx in range(len(ref_cap)):
                    ref_lst = ref_cap[idx].decode().split('|')
                    ref_lst = [r.strip().split(' ') for r in ref_lst if r.strip()]
                    ref_all.append(ref_lst)
                    
                for i in range(output.shape[0]):  # batch_size
                    for j in range(output.shape[1]):
                        hyp_batch[i, j] = output[i, j]
                        if output[i, j] == self.eos_ind:
                            break
                        elif j == output.shape[1] - 1:
                            hyp_batch[i, j] = self.eos_ind 
    
                hyp_batch = hyp_batch.int()
                hyp_all.extend(hyp_batch)

            metrics = decode_output(hyp_all, ref_all, self.idx2word)        
            
            if args.logging == True:
                self.run['Test BLEU-1'].append(metrics['BLEU-1'])
                self.run['Test BLEU-2'].append(metrics['BLEU-2'])
                self.run['Test BLEU-3'].append(metrics['BLEU-3'])
                self.run['Test BLEU-4'].append(metrics['BLEU-4'])
                self.run['Test ROUGE-L'].append(metrics['ROUGE-L'])
                self.run['Test CIDEr'].append(metrics['CIDEr'])
                            
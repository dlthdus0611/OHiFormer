import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default='1', type=str, help='experiment_number')
    parser.add_argument('--experiment', default='Base', type=str)
    parser.add_argument('--tag', default='Default', type=str, help='tag')
    # Path settings
    parser.add_argument('--model_path', type=str, default='results/')    
    # Model parameter settings
    parser.add_argument('--f_dim', type=int, default=128)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--word_emb', type=int, default=300)
    parser.add_argument('--word_hidden_size', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    parser.add_argument('--layer_prepostprocess_dropout', type=float, default=0.2)
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-6)
    # Transformer parameter settings
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_head', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    # Training parameter settings
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optimizer', type=str, default='')
    parser.add_argument('--initial_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # Validation parameter settings
    ## Scheduler
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--warm_epoch', type=int, default=5)  # WarmUp Scheduler
    parser.add_argument('--freeze_epoch', type=int, default=0)
    ### Cosine Annealing
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--tmax', type=int, default = 145)
    ### MultiStepLR
    parser.add_argument('--milestone', type=int, nargs='*', default=[100]) #
    parser.add_argument('--lr_factor', type=float, default=0.1)
    ## etc.
    parser.add_argument('--patience', type=int, default=50, help='Early Stopping')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--re_training_exp', type=str, default=None)
    parser.add_argument('--save_epoch', type=int, default=0)
    parser.add_argument('--server', type=str, default='yj3090')
    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)

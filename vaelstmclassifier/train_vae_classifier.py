import argparse

from vaelstmclassifier.vae_classifier.train \
        import train as vae_classifier_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--batch_size', type=int, default=100,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=88,
                help='input dim')
    parser.add_argument('--intermediate_dim', type=int, default=88,
                help='intermediate dim')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--seq_length', type=int, default=1,
                help='sequence length (concat)')
    parser.add_argument('--class_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--w_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument('--intermediate_class_dim',
                type=int, default=88,
                help='intermediate dims for classes')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--predict_next", action="store_true", 
                help="use x_t to 'autoencode' x_{t+1}")
    parser.add_argument("--use_prev_input", action="store_true",
                help="use x_{t-1} to help z_t decode x_t")
    parser.add_argument('--patience', type=int, default=5,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str,
                default='data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str,
                default='data/input/JSB Chorales_Cs.pickle',
                help='file of training data (.pickle)')
    args = parser.parse_args()
    vae_classifier_train(args)

"""
Main script
"""
import argparse
import os
import random
import sys

import numpy as np

import dynet

from constants import TASK_NAMES, LANGUAGES, EMBEDS, BALANCED, IMBALANCED, SGD, ADAM
from sluice_net import SluiceNetwork, load
import utils


def check_activation_function(arg):
    """Checks allowed argument for --ac option."""
    try:
        functions = [dynet.rectify, dynet.tanh]
        functions = {function.__name__: function for function in functions}
        functions['None'] = None
        return functions[str(arg)]
    except:
        raise argparse.ArgumentTypeError(
            'String {} does not match required format'.format(arg, ))


def main(args):


    train_score = {task: 0 for task in args.task_names}
    dev_score = {task: 0 for task in args.task_names}
    avg_train_score = 0
    avg_dev_score = 0
    
    if args.load:
        assert os.path.exists(args.model_dir),\
            ('Error: Trying to load the model but %s does not exist.' %
             args.model_dir)
        print('Loading model from directory %s...' % args.model_dir)
    
        model_file = None
        params_file = None

        #Load models from different directory based on the type (STSL, MTSL, STML, MTML)
        if(len(args.task_names) ==1):

          if(len(args.languages) == 1):

              model_file = os.path.join(args.model_dir, 'STSL/{}_{}.model'.format(args.languages[0],args.task_names[0]))
              params_file = os.path.join(args.model_dir, 'STSL/{}_{}.pkl'.format(args.languages[0],args.task_names[0]))

          else:

              model_file = os.path.join(args.model_dir, 'STML/{}.model'.format(args.task_names[0]))

              params_file = os.path.join(args.model_dir, 'STML/{}.pkl'.format(args.task_names[0]))

        else:

          if(len(args.languages) ==1):

              model_file = os.path.join(args.model_dir, 'MTSL/{}.model'.format(args.languages[0]))
              params_file = os.path.join(args.model_dir, 'MTSL/{}.pkl'.format(args.languages[0]))
          else:
              model_file = os.path.join(args.model_dir, 'MTML/MTML.model')

              params_file = os.path.join(args.model_dir, 'MTML/MTML.pkl')

        

        model, train_score, dev_score, avg_train_score, avg_dev_score = load(params_file, model_file, args)
        
        if(args.continue_train):#Continue to train the loaded model
          train_score, dev_score, avg_train_score, avg_dev_score= model.fit(args.languages, args.test_languages, args.epochs, args.patience, args.opt, args.threshold,
                    train_dir=args.train_dir, dev_dir=args.dev_dir)#added args.threshold

    else:
        model = SluiceNetwork(args.h_dim,
                              args.h_layers,
                              args.model_dir,
                              args.log_dir,
                              embeds=args.embeds,
                              activation=args.activation,
                              lower=args.lower,
                              noise_sigma=args.sigma,
                              task_names=args.task_names,
                              languages = args.languages,
                              cross_stitch=args.cross_stitch,
                              num_subspaces=args.num_subspaces,
                              constraint_weight=args.constraint_weight,
                              constrain_matrices=args.constrain_matrices,
                              cross_stitch_init_scheme=
                              args.cross_stitch_init_scheme,
                              layer_stitch_init_scheme=
                              args.layer_stitch_init_scheme)
        train_score, dev_score, avg_train_score, avg_dev_score = model.fit(args.languages, args.test_languages, args.epochs, args.patience, args.opt, args.threshold, train_dir=args.train_dir, dev_dir=args.dev_dir)
    


    print('='*50)
    print('Start testing', ','.join(args.test_languages))

    for test_lang in args.test_languages:
      test_X, test_Y, _ = utils.get_data(
              [test_lang], model.task_names, model.word2id,
              model.task2label2id, data_dir=args.test_dir, train=False)

      test_score = model.evaluate(test_X, test_Y, test_lang, args.threshold)




      print('='*50)
      print('\tStart logging {}'.format(test_lang))

   
      utils.log_score(args.log_dir, args.languages, [test_lang], args.task_names, args.embeds, args.h_dim,  args.cross_stitch_init_scheme,
       args.constraint_weight, args.sigma, args.opt, train_score, dev_score, test_score)
      

      print('\tFinished logging{}'.format(test_lang))

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the Sluice Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DyNet parameters

    parser.add_argument('--dynet-autobatch', type=int, #automatically batch some operations to speed up computations 
                    help='use auto-batching (1) (should be first argument)')
    parser.add_argument('--dynet-gpus', type=int,  
                        help='Specify how many GPUs you want to use, if DyNet is compiled with CUDA')
    
    parser.add_argument('--dynet-devices', nargs='+',  choices=['CPU', 'GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'],
                        help='Specify which GPUs do use')
    parser.add_argument('--dynet-seed', type=int, help='random seed for DyNet')
    parser.add_argument('--dynet-mem', type=int, help='memory for DyNet')

    # languages, tasks, and paths
    parser.add_argument('--languages', nargs='+', choices=LANGUAGES, 
                      help='the language datasets to be trained on ')
    
    parser.add_argument('--test-languages', nargs='+', choices=LANGUAGES, 
                      help='the language datasets to be tested on')
    
    parser.add_argument('--train-dir', required=True,
                        help='the directory containing the training data')
    parser.add_argument('--dev-dir', required=True,
                        help='the directory containing the development data')
    parser.add_argument('--test-dir', required=True,
                        help='the directory containing the test data')

    parser.add_argument('--load', action='store_true',
                        help='load the pre-trained model')

    parser.add_argument('--load-action', default='test', 
                        choices=['train', 'test'],
                        help='action after loading the model')

    parser.add_argument('--task-names', nargs='+', default=TASK_NAMES, 
                        choices=TASK_NAMES,
                        help='the names of the tasks (main task is first)')
    parser.add_argument('--model-dir', required=True,
                        help='directory where to save model and param files')
    parser.add_argument('--log-dir', required=True,
                        help='the directory where the results should be logged')
    parser.add_argument('--w-in-dim', type=int, default=64,
                        help='default word embeddings dimension [default: 64]')
    #parser.add_argument('--c-in-dim', type=int, default=100,
    #                    help='input dim for char embeddings [default:100]')
    parser.add_argument('--h-dim', type=int, default=100,
                        help='hidden dimension [default: 100]')
    parser.add_argument('--h-layers', type=int, default=1,
                        help='number of stacked LSTMs [default: 1=no stacking]')
    parser.add_argument('--lower', action='store_true',
                        help='lowercase words (not used)')
    parser.add_argument('--embeds', nargs='?',help='word embeddings file', 
                        choices=EMBEDS, default=None)


    parser.add_argument('--sigma', help='noise sigma', default=0.2, type=float)
    parser.add_argument('--activation', default='tanh',
                        help='activation function [rectify, tanh, ...]',
                        type=check_activation_function)
    parser.add_argument('--opt', '--optimizer', default=SGD,
                        choices=[SGD, ADAM],
                        help='trainer [sgd, adam] default: sgd')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='training epochs [default: 30]')
    parser.add_argument('--patience', default=1, type=int,
                        help='patience for early stopping')

    parser.add_argument('--cross-stitch', action='store_true',
                        help='use cross-stitch units between LSTM layers')



    parser.add_argument('--num-subspaces', default=1, type=int, choices=[1, 2],
                        help='the number of subspaces for cross-stitching; '
                             'only 1 (no subspace) or 2 allowed currently')
    parser.add_argument('--constraint-weight', type=float, default=0.,
                        help='weighting factor for orthogonality constraint on '
                             'cross-stitch subspaces; 0 = no constraint')
    parser.add_argument('--constrain-matrices', type=int, nargs='+',
                        default=[1, 2],
                        help='the indices of the LSTM matrices that should be '
                             'constrained; indices correspond to: Wix,Wih,Wic,'
                             'bi,Wox,Woh,Woc,bo,Wcx,Wch,bc. Best indices so '
                             'far: [1, 2] http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.LSTMBuilder.get_parameter_expressions)')
    parser.add_argument('--cross-stitch-init-scheme', type=str,
                        default=BALANCED, choices=[IMBALANCED, BALANCED],
                        help='which initialisation scheme to use for the '
                             'alpha matrix - currently available: imbalanced '
                             'and balanced (which sets all to '
                             '1/(num_tasks*num_subspaces)). Only available '
                             'with subspaces.')
    parser.add_argument('--layer-stitch-init-scheme', type=str,
                        default=BALANCED,
                        choices=[BALANCED, IMBALANCED],
                        help='initialisation scheme for layer-stitch units; '
                             'default: imbalanced (.9) for last layer weights;'
                             'other choice: balanced (1. / num_layers).')

    parser.add_argument('--threshold', type=float,default=0.5,
                        help='threshold for classfication')
    args = parser.parse_args()
    main(args)

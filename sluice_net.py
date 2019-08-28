#!/usr/bin/env python3
# coding=utf-8
"""
Sluice Network model.
"""


import random
import os
import numpy as np
import pickle
import dynet
from progress.bar import Bar

from predictors import SequencePredictor, Layer, RNNSequencePredictor, \
    BiRNNSequencePredictor, CrossStitchLayer, LayerStitchLayer
from utils import load_embeddings_file, get_data, log_fit, average_by_task, get_label, average_by_lang
from constants import   IMBALANCED, BALANCED, SGD, ADAM, LABELS, MODIFIED_LABELS, TASK_NAMES
from sklearn.metrics import classification_report, f1_score



def load(params_file, model_file, args):
    """
    Loads a model by first initializing a model with the hyperparameters
    and then loading the weights of the saved model.
    :param params_file: the file containing the hyperparameters
    :param model_file: the file containing the weights of the saved model
    :return the loaded AdaptNN model
    """
    params = pickle.load(open(params_file, 'rb'))
    model = SluiceNetwork(params['h_dim'],
                          params['h_layers'],
                          params['model_dir'],
                          params['log_dir'],
                          languages = params['languages'],
                          word2id=params['w2i'],
                          embeds = params['embeds'],
                          activation=params['activation'],
                          task_names=params['task_names'],
                          cross_stitch=params['cross_stitch'],
                          num_subspaces=params['num_subspaces'],
                          constraint_weight=params['constraint_weight'],
                          noise_sigma = params['noise_sigma'],
                          constrain_matrices = params['constrain_matrices'],
                          cross_stitch_init_scheme=params['cross_stitch_init_scheme'],
                          layer_stitch_init_scheme=params['layer_stitch_init_scheme'],
                          best_train_dict = params['best_train_dict'],
                          best_dev_dict = params['best_dev_dict'],
                          avg_train_score = params['avg_train_score'],
                          avg_dev_score = params['avg_dev_score'],
                          best_epoch = params['best_epoch'],
                          oov_id = params['oov_id'])

    model.predictors = model.build_computation_graph()
    
    print('Model loaded from %s...' % model_file, flush=True)
    model.model.populate(model_file)

    return model, params['best_train_dict'], params['best_dev_dict'],  params['avg_train_score'], params['avg_dev_score']
'''
def load_no_args(params_file, model_file):
    """
    Loads a model by first initializing a model with the hyperparameters
    and then loading the weights of the saved model.
    :param params_file: the file containing the hyperparameters
    :param model_file: the file containing the weights of the saved model
    :return the loaded AdaptNN model
    """
    params = pickle.load(open(params_file, 'rb'))
    model = SluiceNetwork(params['h_dim'],
                          params['h_layers'],
                          params['model_dir'],
                          params['log_dir'],
                          languages = params['languages'],
                          word2id=params['w2i'],
                          embeds = params['embeds'],
                          activation=params['activation'],
                          task_names=params['task_names'],
                          cross_stitch=params['cross_stitch'],
                          num_subspaces=params['num_subspaces'],
                          constraint_weight=params['constraint_weight'],
                          noise_sigma = params['noise_sigma'],
                          constrain_matrices = params['constrain_matrices'],
                          cross_stitch_init_scheme=params['cross_stitch_init_scheme'],
                          layer_stitch_init_scheme=params['layer_stitch_init_scheme'],
                          best_train_dict = params['best_train_dict'],
                          best_dev_dict = params['best_dev_dict'],
                          avg_train_score = params['avg_train_score'],
                          avg_dev_score = params['avg_dev_score'],
                          best_epoch = params['best_epoch'],
                          oov_id = params['oov_id'])

    model.predictors = model.build_computation_graph()
    
    print('Model loaded from %s...' % model_file, flush=True)
    model.model.populate(model_file)
    return model, params['best_train_dict'], params['best_dev_dict'],  params['avg_train_score'], params['avg_dev_score']
'''

class SluiceNetwork(object):
    def __init__(self, h_dim, h_layers, model_dir, log_dir, task_names, languages,
                 embeds=None, activation=dynet.tanh, lower=False,
                 noise_sigma=0.1, cross_stitch=False, num_subspaces=1, 
                 constraint_weight=0,  constrain_matrices=[1, 2], cross_stitch_init_scheme=IMBALANCED,  
                   layer_stitch_init_scheme=BALANCED, best_train_dict = {}, best_dev_dict = {},
                    avg_train_score=0, avg_dev_score =0, best_epoch=-1, word2id={}, oov_id = None):
        """
        :param h_dim: The hidden dimension of the model.
        :param h_layers: The number of hidden layers.
        :param model_dir: The directory where the model should be saved
        :param log_dir: The directory where the log should be saved
        :param task_names: the names of the tasks
        :param langauges: the training languages of the model 
        :param embeds: the pre-trained embedding used by the model
        :param activation: the DyNet activation function that should be used
        :param lower: whether the words should be lower-cased
        :param noise_sigma: the stddev of the Gaussian noise that should be used
                            during training if > 0.0
        :param cross_stitch: whether to use cross-stitch units

        :param num_subspaces: the number of subspaces to use (1 or 2)
        :param constraint_weight: weight of subspace orthogonality constraint
                                  (default: 0 = no constraint)
        :param constrain_matrices: indices of LSTM weight matrices that should
                                   be constrained (default: [1, 2])
        :param cross_stitch_init_scheme: initialisation scheme for cross-stitch
        :param layer_stitch_init_scheme: initialisation scheme for layer-stitch

        :param  best_train_dict: dictionary storing the best scores on training set
        :param  best_dev_dict: dictionary storing the best scores on development set         
        :param  avg_train_score: best unweighted average training score over all tasks and all metrics
        :param  avg_dev_score: best unweighted average development score over all tasks and all metrics
        :param  best_epoch: the epoch of the best performance
        :param  word2id: dictionary storing the words to the idx of the word embedding
        :param  oov_id: the idx of the word which do not appear in the pre-trained word embedding


        """
        self.word2id = word2id  

        self.task_names = task_names
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.w_in_dim = 0


        if(len(task_names) ==1):

            if(len(languages) == 1):
                task_directory = os.path.join(model_dir,'STSL/')
                if not os.path.exists(task_directory):
                    os.mkdir(task_directory)
                self.model_file = os.path.join(model_dir, 'STSL/{}_{}.model'.format(languages[0],task_names[0]))
                self.params_file = os.path.join(model_dir, 'STSL/{}_{}.pkl'.format(languages[0],task_names[0]))
            else:
                task_directory = os.path.join(model_dir,'STML/')
                if not os.path.exists(task_directory):
                    os.mkdir(task_directory)
                self.model_file = os.path.join(model_dir, 'STML/{}.model'.format(task_names[0]))

                self.params_file = os.path.join(model_dir, 'STML/{}.pkl'.format(task_names[0]))
                

        else:

            if(len(languages) ==1):
                task_directory = os.path.join(model_dir,'MTSL/')
                if not os.path.exists(task_directory):
                    os.mkdir(task_directory)
                self.model_file = os.path.join(model_dir, 'MTSL/{}.model'.format(languages[0]))
                self.params_file = os.path.join(model_dir, 'MTSL/{}.pkl'.format(languages[0]))
               
            else:
                task_directory = os.path.join(model_dir,'MTML/')
                if not os.path.exists(task_directory):
                    os.mkdir(task_directory)
                self.model_file = os.path.join(model_dir, 'MTML/MTML.model')
                self.params_file = os.path.join(model_dir, 'MTML/MTML.pkl')


        self.cross_stitch = cross_stitch
        self.num_subspaces = num_subspaces
        self.constraint_weight = constraint_weight
        self.constrain_matrices = constrain_matrices
        self.cross_stitch_init_scheme = cross_stitch_init_scheme
        self.layer_stitch_init_scheme = layer_stitch_init_scheme
        self.model = dynet.Model()  # init model
        # term to capture sum of constraints over all subspaces
        self.subspace_penalty = self.model.add_parameters(
            1, init=dynet.NumpyInitializer(np.zeros(1)))
        # weight of subspace constraint
        self.constraint_weight_param = self.model.add_parameters(
            1, init=dynet.NumpyInitializer(np.array(self.constraint_weight)))


        task2label2id = {}

        for task in task_names:
            labels = LABELS[task]#TO BE CHANGED AGAIN to MODIFIED_LABELS[task] 
            task2label2id[task] = {}
            count = 0

            for label in LABELS[task]:
                task2label2id[task][label] = count
                count+=1


        
        self.task2label2id = task2label2id  # need one dictionary per task

        self.languages = languages
        self.h_dim = h_dim
        self.activation = activation
        self.lower = lower
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {}
        self.wembeds = None  # lookup: embeddings for words
        self.embeds = embeds
 
        self.best_train_dict = best_train_dict 
        self.best_dev_dict = best_dev_dict

        self.best_epoch = best_epoch

        self.avg_train_score  = avg_train_score 
        self.avg_dev_score = avg_dev_score
        self.oov_id = oov_id

    def save(self):
        """Save model. DyNet only saves parameters. Save rest separately."""
        self.model.save(self.model_file)
        myparams = {"task_names": self.task_names,
                    "languages": self.languages,
                    "w2i": self.word2id,
                    "task2tag2idx": self.task2label2id,
                    "activation": self.activation,
                    "h_dim": self.h_dim,
                    "h_layers": self.h_layers,
                    "embeds": self.embeds,
                    'model_dir': self.model_dir,
                    'cross_stitch': self.cross_stitch,
                    'num_subspaces': self.num_subspaces,
                    'constraint_weight': self.constraint_weight,
                    'cross_stitch_init_scheme': self.cross_stitch_init_scheme,
                    'layer_stitch_init_scheme': self.layer_stitch_init_scheme,
                    'constrain_matrices': self.constrain_matrices,
                    'noise_sigma': self.noise_sigma,
                    'best_train_dict': self.best_train_dict,
                    'best_dev_dict': self.best_dev_dict,
                    'best_epoch': self.best_epoch,
                    'oov_id': self.oov_id,
                    'log_dir': self.log_dir,
                    'avg_train_score': self.avg_train_score,
                    'avg_dev_score':self.avg_dev_score }
        pickle.dump(myparams, open(self.params_file, "wb"))




    def build_computation_graph(self):
        """Builds the computation graph."""
        # initialize the word embeddings using the pre-trained embedding file

        embeddings, emb_dim = load_embeddings_file(self.embeds, self.languages,
                                       lower=self.lower)
        self.w_in_dim = emb_dim

        num_words = len(set(embeddings.keys()).union(set(self.word2id.keys())))
        self.wembeds = self.model.add_lookup_parameters((num_words, emb_dim))
        self.oov_id = set(range(num_words))

        #Find words which do not appear in the pre-trained embeddings
        #by removing words which have appeared
        for i, word in enumerate(embeddings.keys()):
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id.keys())
            self.wembeds.init_row(self.word2id[word], embeddings[word])

            self.oov_id.remove(self.word2id[word])
       

        layers = []  # inner layers


        output_layers_dict = {}  # from task_name to actual predictor


        # we have a separate layer for each task for cross-stitching;
        # otherwise just 1 layer for all tasks with hard parameter sharing
        num_task_layers = len(self.task_names) if self.cross_stitch else 1
        #print("task names")
        #print(self.task_names)
        #print("num_task_layers:")
        #print(len(self.task_names))
        cross_stitch_layers = []


        for layer_num in range(self.h_layers):
            print(">>> %d layer_num" % layer_num, flush=True)
            input_dim = self.w_in_dim if layer_num == 0 \
                else self.h_dim
  
            task_layers = []
            # get one layer per task for cross-stitching or just one layer
            for task_id in range(num_task_layers):
                builder = dynet.LSTMBuilder(1, input_dim, self.h_dim, self.model)
                task_layers.append(BiRNNSequencePredictor(builder))
            layers.append(task_layers)
            if self.cross_stitch:
                print('Using cross-stitch units after layer %d...' % layer_num,
                      flush=True)
                cross_stitch_layers.append(
                    CrossStitchLayer(self.model, len(self.task_names),
                                     self.h_dim, self.num_subspaces,
                                     self.cross_stitch_init_scheme))

        layer_stitch_layers = []


        # store at which layer to predict task
        for task_name in self.task_names:
            task_num_labels = len(self.task2label2id[task_name])

            print('Using an MLP for task losses.', flush=True)

            input_dim = self.h_dim * 2
            activation = dynet.softmax

            layer_output = None
            if('sentiment' in task_name):#Multi-label classification
            #use one binary classification layer for each label
                layer_output =[]
                for _ in range(task_num_labels):
                    layer_output.append(Layer(self.model, input_dim, 2,
                                 activation, mlp=True))

            else:
                layer_output = Layer(self.model, input_dim, task_num_labels,
                                 activation, mlp=True)

            output_layers_dict[task_name] = layer_output#sequence_predictor

            if(self.h_layers > 1): 
                # w/o cross-stitching, we only use one LayerStitchLayer
                layer_stitch_layers.append(
                    LayerStitchLayer(self.model, self.h_layers, self.h_dim,
                                     self.layer_stitch_init_scheme))

        print('#\nOutput layers: %d\n' % len(output_layers_dict), flush=True)


        predictors = dict()
        predictors["inner"] = layers
        predictors['cross_stitch'] = cross_stitch_layers
        predictors['layer_stitch'] = layer_stitch_layers
        predictors["output_layers_dict"] = output_layers_dict
        return predictors





    def fit(self, train_languages, test_lang, num_epochs, patience, optimizer, threshold, train_dir,
            dev_dir):
        """
        Train the model, return the train and dev score
        :param train_language: the language used for training
        :param num_epochs: the max number of epochs the model should be trained
        :param patience: the patience to use for early stopping
        :param optimizer: the optimizer that should be used
        :param train_dir: the directory containing the training files
        :param dev_dir: the directory containing the development files
        :param threshold added

        """

        first_train = True if self.best_epoch==(-1) else False#Check whether this is a loaded model

        print("Reading training data from %s..." % train_dir, flush=True)
        train_X, train_Y, word2id = get_data(train_languages, self.task_names, word2id=self.word2id, task2label2id=self.task2label2id, 
            data_dir=train_dir, train=first_train)
        print("Finished reading training data")


        print("Reading development data from %s..." % train_dir, flush=True)
        dev_X, dev_Y, _ = get_data(train_languages, self.task_names, word2id, self.task2label2id, 
         data_dir=dev_dir, train=False)
        print("Finished reading development data")

        print('Length of training data:', len(train_X), flush=True)
        print('Length of development data:', len(dev_X), flush=True)


        if(first_train):
            self.word2id = word2id
            

            print('Building the computation graph...', flush=True)
            self.predictors= \
                self.build_computation_graph()

        if optimizer == SGD:
            trainer = dynet.SimpleSGDTrainer(self.model)
        elif optimizer == ADAM:
            trainer = dynet.AdamTrainer(self.model)
        else:
            raise ValueError('%s is not a valid optimizer.' % optimizer)

        train_data = list(zip(train_X, train_Y))

        num_iterations = 0
        num_epochs_no_improvement = 0


        train_score = {}
        dev_score = {}

        print('Training model with %s for %d epochs and patience of %d.'
              % (optimizer, num_epochs, patience))

        for epoch in range(self.best_epoch+1, num_epochs):
            
            print('', flush=True)

            bar = Bar('Training epoch %d/%d...' % (epoch+1, num_epochs),
                      max=len(train_data), flush=True)

            # keep track of the # of updates, total loss, and total # of
            # predicted instances per task
            task2num_updates = {task: 0 for task in self.task_names}
            task2total_loss = {task: 0.0 for task in self.task_names}
            task2total_predicted = {task: 0.0 for task in self.task_names}
            total_loss = 0.0
            total_penalty = 0.0
            total_predicted = 0.0

            random.shuffle(train_data)

            # for every instance, we optimize the loss of the corresponding task
            for word_indices, task2label_id_seq in train_data:
                # get the concatenated word and char-based features for every
                # word in the sequence
                features = self.get_word_features(word_indices)

                for task, y in task2label_id_seq.items():

                    output, penalty = self.predict(features, task, train=True)
         
                    
                    if task not in TASK_NAMES:
                        raise NotImplementedError('Task %s has not been '
                                                  'implemented yet.' % task)


                    loss = dynet.esum([pick_neg_log(o, gold) for \
                        o, gold in zip(output, y)])

                    lv = loss.value()


                    # sum the loss and the subspace constraint penalty
            
                    combined_loss = loss + dynet.const_parameter(self.constraint_weight_param) * penalty

                    total_loss += lv
                    total_penalty += penalty.value()
                    total_predicted += 1
                    task2total_loss[task] += lv
                    task2total_predicted[task] += 1
                    task2num_updates[task] += 1

                    # back-propagate through the combined loss
                    combined_loss.backward()
                    trainer.update()
                bar.next()
                num_iterations += 1

            print("\nEpoch %d. Loss per instance: %.3f. Penalty per instance: %.3f. "
                  % (epoch+1, total_loss / total_predicted,
                     total_penalty / total_predicted), end='', flush=True)

            print('Loss per instance by task: ')

            for task in task2total_loss.keys():
                print('%s: %.3f. ' % (task, task2total_loss[task] /
                                      task2total_predicted[task]),
                      end='', flush=True)
            print('', flush=True)

            

            # evaluate after every epoch

            avg_train_score_by_task_list = []#Each item stores the avg train score (by task) for a particular language
            avg_dev_score_by_task_list = []#Each item stores the avg dev score (by task) for a particular language
            train_data_size_list = []#Each item stores the size for a particular language train set
            dev_data_size_list = []#Each item stores the size for a particular language dev set

            for lang in train_languages:
                #changed utils.get_data( 
                #changed model to self everywhere,checkif it has to be replaced by self.model
                #changed args.train_dir to train_dir
                train_eval_X, train_eval_Y, _ = get_data(
                          [lang], self.task_names, self.word2id,
                          self.task2label2id, data_dir=train_dir, train=False)


                train_data_size_list+=[len(train_eval_Y)]

                #changed args.dev_dir to dev_dir
                dev_eval_X, dev_eval_Y, _ = get_data(
                          [lang], self.task_names, self.word2id,
                          self.task2label2id, data_dir= dev_dir, train=False)


                dev_data_size_list+=[len(dev_eval_Y)]


                #changed args.threshold to threshold
                train_score = self.evaluate(train_eval_X, train_eval_Y, lang, threshold)

                #changed args.threshold to threshold
                dev_score = self.evaluate(dev_eval_X, dev_eval_Y, lang, threshold)
                #changed utils.average_by_task
                avg_train_score_by_task_list.append(average_by_task(train_score))
                avg_dev_score_by_task_list.append(average_by_task(dev_score))



                print('='*50)
                print('\tStart logging for {} in epoch {}'.format(test_lang, epoch+1))

                #changed  utils.log_fit
                log_fit(self.log_dir, epoch+1, train_languages, test_lang, self.task_names, train_score, dev_score)


                print('\tFinish logging for {} in epoch {}'.format(test_lang, epoch+1))



            #Compute the weighted average over all languages and use it to determine the overall performance of training
            total_train_size = len(train_Y)
            total_dev_size = len(dev_Y)

            #changed util.average_by_lang
            avg_train_score = average_by_lang(avg_train_score_by_task_list, train_data_size_list, 
                total_train_size)

            #changed util.average_by_lang
            avg_dev_score = average_by_lang(avg_dev_score_by_task_list, dev_data_size_list, 
                total_dev_size)
       
            if avg_dev_score > self.avg_dev_score:

                self.avg_dev_score = avg_dev_score
                self.avg_train_score = avg_train_score

                self.best_train_dict = train_score
                self.best_dev_dict = dev_score


                self.best_epoch = epoch
                num_epochs_no_improvement = 0
                print('Saving model to directory %s...' % self.model_dir,
                      flush=True)
                self.save()
            else:

                num_epochs_no_improvement += 1



            if num_epochs_no_improvement == patience:
                #dynet.load(self.model_file, self.model)
                break


        print('Finished training', flush=True)
        print('Loading the best performing model from %s...'\
                      % self.model_dir, flush=True)

        self.model.populate(self.model_file)


        
        return self.best_train_dict, self.best_dev_dict, self.avg_train_score, self.avg_dev_score
       


    def predict(self, features, task_name, train=False):
        """
        Steps through the computation graph and obtains predictions for the
        provided input features.
        :param features: a list of word  embeddings for every word in the sequence
        :param task_name: the name of the task that should be predicted
        :param train: if the model is training; apply noise in this case
        :return output: the output predictions
                penalty: the summed subspace penalty (0 if no constraint)
        """
        if train:  # noise is added only at training time

            features = [dynet.noise(fe, self.noise_sigma) for fe in
                        features]



        # only if we use cross-stitch we have a layer for each task;
        # otherwise we just have one layer for all tasks
        num_layers = self.h_layers
        inputs = [features] * len(self.task_names)
        inputs_rev = [features] * len(self.task_names)

        target_task_id = self.task_names.index(
            task_name) if self.cross_stitch else 0
        
        #added
        num_task_layers = len(self.task_names) if self.cross_stitch else 1
    

        # collect the forward and backward sequences for each task at every
        # layer for the layer connection units
        layer_forward_sequences = []
        layer_backward_sequences = []

        penalty = dynet.const_parameter(self.subspace_penalty)

        for i in range(0, num_layers):
            forward_sequences = []
            backward_sequences = []
            for j in range(num_task_layers):
                predictor = self.predictors['inner'][i][j]
                forward_sequence, backward_sequence = predictor.predict_sequence(
                    inputs[j], inputs_rev[j])
                if i > 0 and self.activation:
                    # activation between LSTM layers
                    forward_sequence = [self.activation(s) for s in
                                        forward_sequence]
                    backward_sequence = [self.activation(s) for s in
                                         backward_sequence]
                forward_sequences.append(forward_sequence)
                backward_sequences.append(backward_sequence)

                if self.num_subspaces == 2 and self.constraint_weight != 0:
                    # returns a list per layer, i.e. here a list with one item
                    lstm_parameters = \
                        predictor.builder.get_parameter_expressions()[0]



                    # lstm parameters consists of these weights:
                    # Wix,Wih,Wic,bi,Wox,Woh,Woc,bo,Wcx,Wch,bc
                    for param_idx in range(len(lstm_parameters)):
                        if param_idx in self.constrain_matrices:
                            W = lstm_parameters[param_idx]
                            W_shape = np.array(W.value()).shape

                            if(len(W_shape) <2):
                                W_shape = [W_shape[0], 1]

                            # split matrix into its two subspaces
                            W_subspaces = dynet.reshape(W, (
                                self.num_subspaces, W_shape[0] / float(
                                    self.num_subspaces), W_shape[1]))
                            subspace_1, subspace_2 = W_subspaces[0], W_subspaces[1]

                            # calculate the matrix product of the two matrices
                            matrix_product = dynet.transpose(
                                subspace_1) * subspace_2

                            # take the squared Frobenius norm by squaring
                            # every element and then summing them
                            squared_frobenius_norm = dynet.sum_elems(
                                dynet.square(matrix_product))
                            penalty += squared_frobenius_norm

            if self.cross_stitch:
                # takes as input a list of input lists and produces a list of
                # outputs where the index indicates the task
                forward_sequences = self.predictors['cross_stitch'][
                    i].stitch(forward_sequences)
                backward_sequences = self.predictors['cross_stitch'][
                    i].stitch(backward_sequences)

            inputs = forward_sequences
            inputs_rev = backward_sequences
            layer_forward_sequences.append(forward_sequences)
            layer_backward_sequences.append(backward_sequences)

            if i == num_layers-1:
                output_predictor = \
                    self.predictors['output_layers_dict'][task_name]

                # get the forward/backward states of all task layers
                task_forward_sequences = [
                    layer_seq_list[target_task_id][-1] for
                    layer_seq_list in layer_forward_sequences]

                task_backward_sequences = [
                    layer_seq_list[target_task_id][0] for
                    layer_seq_list in layer_backward_sequences]


                if(num_layers > 1):
                    forward_input = \
                        self.predictors['layer_stitch'][
                            target_task_id].stitch(task_forward_sequences)
                    backward_input = \
                        self.predictors['layer_stitch'][
                            target_task_id].stitch(task_backward_sequences)


                else:
                    forward_input = task_forward_sequences[0]
                    backward_input = task_backward_sequences[0]
                


               
                concat_layer = dynet.concatenate([forward_input, backward_input])

                if train and self.noise_sigma > 0.0:
                    concat_layer = dynet.noise(concat_layer, self.noise_sigma)
                                    
                output = []

                if('sentiment' in task_name):#Multi-label
                    
                    for i in range(len(output_predictor)):

                        output.append(output_predictor[i](concat_layer))


                else:
                    output.append(output_predictor(concat_layer))
                

                #output = output_predictor.predict_sequence(concat_layer)

                return output, penalty
        raise Exception('Error: This place should not be reached.')



    def evaluate(self, test_X, test_Y, test_lang, threshold):
        """
        Computes accuracy on a test file.
        :param test_X: the test data; a list of (word_ids, char_ids) tuples
        :param test_Y: labels; a list of task-to-label sequence mappings
        :param test_lang: language of the test data
        :param threshold: threshold for classification in multi-label prediction

        :return a dictionary storing the macro-f1 and micro-f1 scores of all tasks
        """
        dynet.renew_cg(immediate_compute = True)#(immediate_compute = True, check_validity = True) #is_valid not yet implemented for CUDA

        #Display the parameters
        '''
        if self.cross_stitch:
            for layer_num in range(self.h_layers):
                alphas = dynet.parameter(
                    self.predictors['cross_stitch'][layer_num].alphas).value()
                print('Cross-stitch unit values at layer %d.' % layer_num,
                      end=' ', flush=True)
                if self.num_subspaces > 1:
                    print(np.array(alphas).flatten())
                else:
                    for i, task_i in enumerate(self.task_names):
                        for j, task_j in enumerate(self.task_names):
                            print('%s-%s: %3f.' % (task_i, task_j,
                                                   alphas[i][j]),
                                  end=' ', flush=True)
                print('')


        '''

        y_true_dict = {task: [] for task in self.task_names}

        y_pred_dict = {task: [] for task in self.task_names}

                      
        for i, (word_indices, task2label_id_seq)\
                in enumerate(zip(test_X, test_Y)):
            for task, label_id_seq in task2label_id_seq.items():
                features = self.get_word_features(word_indices)
                output, _ = self.predict(features, task, train=False)
                
                y_true_dict[task].append(label_id_seq)

                if('sentiment' in task):#Multi-label classification
                    output_seq = []
  

                    for o in output:
                        o_val = o.value()    

                        if(o_val[1]>=threshold):
                            output_seq.append(1)
                        else:
                            output_seq.append(0)

                    y_pred_dict[task].append(output_seq)
                else:
                    y_pred_dict[task].append([np.argmax(o.value()) for o in output])





        res_dict = {}
        for task in self.task_names:

            res_dict[task] = {'micro_f1': 0, 'macro_f1': 0}


        for task in y_true_dict:


            clf_dict = classification_report(np.array(y_true_dict[task]), np.array(y_pred_dict[task]),
                    output_dict=True)
            precision = clf_dict['micro avg']['precision']
            recall = clf_dict['micro avg']['recall'] 
            
            divisor = precision + recall
            if divisor<0.000001:
                divisor = 0.000001

            if(divisor > 0):

                res_dict[task]['micro_f1'] = (2*precision*recall)\
                /(divisor)
            else: #changing dvisor
                print("The sum of precision and recall equals zero.")
                res_dict[task]['micro_f1'] = 0

            precision = clf_dict['macro avg']['precision']
            recall = clf_dict['macro avg']['recall'] 
            
            if divisor<0.000001:
                divisor = 0.000001

            if(divisor > 0):
                res_dict[task]['macro_f1'] = (2*precision*recall)\
                /(divisor)
            else:
                print("The sum of precision and recall equals zero.")
                res_dict[task]['macro_f1'] = 0
        print(test_lang)
        print(threshold)
        print(res_dict)
        return res_dict
   




    def get_word_features(self, word_indices):
        """
        Produce word and character features that can be used as input for the
        predictions.
        :param word_indices: a list of word indices
        :return: a list of word embeddigs
        """
        dynet.renew_cg(immediate_compute = True)#(immediate_compute = True, check_validity = True)  # new graph #is_valid() not implemented for CUDA yet

        features = []

        for w_idx in word_indices:
            update_flag = False
            if(w_idx in self.oov_id):
                #Allow the vocabs which are not in pre-load embeddings to
                #be updated during training
                update_flag = True 

            embed_vec = dynet.lookup(self.wembeds,index=w_idx, update=update_flag)
            features.append(embed_vec)
        
        return features


def pick_neg_log(pred, gold):
    """Get the negative log-likelihood of the predictions."""
    return -dynet.log(dynet.pick(pred, gold))

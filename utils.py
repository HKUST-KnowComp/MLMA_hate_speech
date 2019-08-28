"""
Utility methods for data processing.
"""
import os
from glob import glob
import itertools
import csv
import pandas as pd
from constants import NUM, NUMBERREGEX, UNK, WORD_START, WORD_END, EMBEDS_FILES, FULL_LANG, LABELS, MODIFIED_LABELS

def print_task_labels(task_name, label2id, id_sequence, file): 
    #Convert label_id sequence to label sequence and write to file
    #changed the original function completely
    with open(file, 'a+') as f:
        writer = csv.writer(f,delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', task_name])
        label_list=dict()
        for task_id, labels_ids in label2id.items():
            if task_name==task_id:
                for label, idx in labels_ids.items():
                    label_list[label] = idx
            #print(label_list)
            
        count = 1
        #with open(file, 'a+') as f:
        for label_idx_seq in id_sequence:
            #Create a label_sequence for each tweet
            label_seq = []
            for task, label_idx in label_idx_seq.items():
                #intialize_values
                #target_val=''
                #group_val=''
                #annotator_val=[]
                #sentiment_val=[]
                #Non multilabel_tasks, labels are of the form [1, [7], [12], ...
                if task==task_name:
                    if task=='target' or task =='group' or task=='directness':
                        for target_label, indice in label2id[task].items(): 
                            if indice==label_idx[0]:
                                if task=='target':
                                    val=target_label
                                else:
                                    val=target_label
                #Multilabel tasks, labels are of the form [1, 0, 0, 1, 0, 0], ... such that each column represents one label
                    elif task=='annotator_sentiment':
                        val=[]
                        for j in range(len(label_idx)):
                            if label_idx[j]>0:
                                for label, indice in label2id[task].items():
                                #if labels[j]==1 or label number j ==1 append the name of the label
                                    if indice==j:
                                        val.append(label)
                    elif task=='sentiment':
                        val=[]
                        for j in range(len(label_idx)):
                            if label_idx[j]>0:
                                for label, indice in label2id[task].items():
                                    #if labels[j]==1 or label number j ==1 append the name of the label
                                    if indice==j:
                                        val.append(label)
                    writer.writerow([count,val])
                    count+=1
            #target_val=''
            #group_val=''
            #annotator_val=[]
            #sentiment_val=[]
                

    f.close()




#write functions for studying correlations
def save_generated_labels_in_csv_file(label2id, id_sequence, file): 
    #Convert label_id sequence to label sequence and write to file
    #changed the original function completely
    with open(file, 'a+') as f:
        writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID','annotator_sentiment','sentiment','group','target'])
        label_list=dict()
        for task, labels_ids in label2id.items():
        #print (task)
            for label, idx in labels_ids.items():
                label_list[label] = idx
            #print(label_list)
            
        count = 1
        #with open(file, 'a+') as f:
        for label_idx_seq in id_sequence:
            #Create a label_sequence for each tweet
            label_seq = []
            for task, label_idx in label_idx_seq.items():
                #intialize_values
                #target_val=''
                #group_val=''
                #annotator_val=[]
                #sentiment_val=[]
                #Non multilabel_tasks, labels are of the form [1, [7], [12], ...
                if task=='target' or task =='group':
                    for target_label, indice in label2id[task].items(): 
                        if indice==label_idx[0]:
                            if task=='target':
                                target_val=target_label
                            else:
                                group_val=target_label
                #Multilabel tasks, labels are of the form [1, 0, 0, 1, 0, 0], ... such that each column represents one label
                elif task=='annotator_sentiment':
                    annotator_val=[]
                    for j in range(len(label_idx)):
                        if label_idx[j]>0:
                            for label, indice in label2id[task].items():
                            #if labels[j]==1 or label number j ==1 append the name of the label
                                if indice==j:
                                    annotator_val.append(label)
                elif task=='sentiment':
                    sentiment_val=[]
                    for j in range(len(label_idx)):
                        if label_idx[j]>0:
                            for label, indice in label2id[task].items():
                                #if labels[j]==1 or label number j ==1 append the name of the label
                                if indice==j:
                                    sentiment_val.append(label)
            writer.writerow([count,sentiment_val,target_val,group_val,annotator_val])
            #target_val=''
            #group_val=''
            #annotator_val=[]
            #sentiment_val=[]
            count+=1

    f.close()



def get_label(label2id, id_sequence, file): 
    #Convert label_id sequence to label sequence and write to file
    #changed the original function completely
    label_list=dict()
    for task, labels_ids in label2id.items():
        #print (task)
        for label, idx in labels_ids.items():
            label_list[label] = idx
            #print(label_list)
            

    count = 1
    with open(file, 'a+') as f:
        for label_idx_seq in id_sequence:
            #Create a label_sequence for each tweet
            label_seq = []
            for task, label_idx in label_idx_seq.items():
                #Non multilabel_tasks, labels are of the form [1, [7], [12], ...
                if task=='target' or task =='group':
                    for target_label, indice in label2id[task].items(): 
                        if indice==label_idx[0]:
                            label_seq.append(target_label)
                #Multilabel tasks, labels are of the form [1, 0, 0, 1, 0, 0], ... such that each column represents one label
                elif task=='annotator_sentiment' or task =='sentiment':
                    for j in range(len(label_idx)):
                        if label_idx[j]>0:
                            for label, indice in label2id[task].items():
                                #if labels[j]==1 or label number j ==1 append the name of the label
                                if indice==j:
                                    label_seq.append(label)
            f.write(str(count) +'.\t'+','.join(label_seq) +'\n')
            count+=1

        f.close()

def normalize(word):
    """Normalize a word by lower-casing it or replacing it if it is a number."""
    return NUM if NUMBERREGEX.match(word) else word.lower()

def average_by_task(score_dict): 
#Compute unweighted average of all metrics among all tasks
    total = 0
    count = 0

    for key in score_dict:
     
        total+=(score_dict[key]['micro_f1'] + score_dict[key]['macro_f1'])
        count+=2


    return total/float(count)

def average_by_lang(score_list, data_size_list, total_data_size): 
    #Compute weighted average of all languages
    res = 0

    for idx in range(len(score_list)):
        ratio = float(data_size_list[idx]) / total_data_size
        res += ratio * score_list[idx]

    return res

def load_embeddings_file(embeds, languages, sep=" ", lower=False):
    """Loads a word embedding file."""


    embed_dir = EMBEDS_FILES[embeds]
    file_name_list = []
    for f in os.listdir(embed_dir):
        if (any([f.endswith(lang+'.vec') for lang in languages])):
            file_name_list.append(os.path.join(embed_dir,f))


    word2vec = {}
    total_num_words = 0
    embed_dim = 0
    encoding = None
    for file_name in file_name_list:
        print('\n\n Loading {}.....\n\n'.format(file_name))
        if(file_name.endswith('ar.vec') or file_name.endswith('fr.vec')):
            encoding='utf-8'
        with open(file=file_name, mode='r', encoding=encoding) as f:
            (num_words, embed_dim) = (int(x) for x in f.readline().rstrip('\n').split(' '))
            total_num_words+=num_words
            for idx, line in enumerate(f):
                if((idx+1)%(1e+5)==0):
                    print('Loading {}/{} words'.format(idx+1, num_words))
                fields = line.rstrip('\n').split(sep)
                vec = [float(x) for x in fields[1:]]
                word = fields[0]
                if lower:
                    word = word.lower()
                word2vec[word] = vec
    print('Loaded pre-trained embeddings of dimension: {}, size: {}, lower: {}'
          .format(embed_dim, total_num_words, lower))
    return word2vec, embed_dim






def get_data(languages, task_names, word2id=None, task2label2id=None, data_dir=None,
         train=True, verbose=False):
    """
    :param languages: a list of languages from which to obtain the data
    :param task_names: a list of task names
    :param word2id: a mapping of words to their ids
    :param char2id: a mapping of characters to their ids
    :param task2label2id: a mapping of tasks to a label-to-id dictionary
    :param data_dir: the directory containing the data
    :param train: whether data is used for training (default: True)
    :param verbose: whether to print more information re file reading
    :return X: a list of tuples containing a list of word indices and a list of
               a list of character indices;
            Y: a list of dictionaries mapping a task to a list of label indices;
            org_X: the original words; a list of lists of normalized word forms;
            org_Y: a list of dictionaries mapping a task to a list of labels;
            word2id: a word-to-id mapping;
            char2id: a character-to-id mapping;
            task2label2id: a dictionary mapping a task to a label-to-id mapping.
    """
    X = []
    Y = []
    org_X = []
    org_Y = []

    # for training, we initialize all mappings; for testing, we require mappings
    if train:
 
        # create word-to-id, character-to-id, and task-to-label-to-id mappings
        word2id = {}


        # set the indices of the special characters
        word2id[UNK] = 0  # unk word / OOV


    for language in languages:
        num_sentences = 0
        num_tokens = 0

        full_lang = FULL_LANG[language]
        #file_reader = iter(())
        language_path = os.path.join(data_dir, full_lang)


        assert os.path.exists(language_path), ('language path %s does not exist.'
                                             % language_path)

        csv_file = os.path.join(language_path,os.listdir(language_path)[0])

        df = pd.read_csv(csv_file)


        #Column headers are HITId, tweet, sentiment, directness, annotator_sentiment, target, group

        for index, instance in df.iterrows():
            num_sentences+=1
            #sentence = instance['tweet'].split()
            sentence = instance['tweet'].split()

            sentence_word_indices = []  # sequence of word indices
            sentence_char_indices = []  # sequence of char indice

            # keep track of the label indices and labels for each task
            sentence_task2label_indices = {}

            for i, word in enumerate(sentence):
                num_tokens+=1

                if train and word not in word2id:
                    word2id[word] = len(word2id)

                sentence_word_indices.append(word2id.get(word, word2id[UNK]))

        


            labels = None

            for task in task2label2id.keys():
                if('sentiment' in task):
                  labels = instance[task].split('_')
                else:
                  labels = [instance[task]]
                
                if('sentiment' in task):#Multi-label

                    sentence_task2label_indices[task]=[0]*len(task2label2id[task])

                    for label in labels:
                        label_idx = task2label2id[task][label]
                        sentence_task2label_indices[task][label_idx]=1


                else:

                    sentence_task2label_indices[task] = [task2label2id[task][labels[0]]]


            X.append(sentence_word_indices)
            Y.append(sentence_task2label_indices)

    assert len(X) == len(Y)
    return X, Y, word2id
      


#Log the training process

def log_fit(log_dir, epoch, languages, test_lang, task_names, train_score, dev_score):
    if(len(task_names) ==1):
        task_name = task_names[0]

        if(len(languages) == 1):
            task_directory = os.path.join(log_dir,'STSL/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'STSL/{}_{}.csv'.format(languages[0],task_names[0]))

        else:
            task_directory = os.path.join(log_dir,'STML/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'STML/{}.csv'.format(task_names[0]))

       #This function needs to be changed
        if(os.path.exists(file)):
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)


            writer.writerow([epoch, test_lang, train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                    dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1']])                        
        
        else:
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(['epoch',  'test_lang', task_name+'-train-micro-f1',  task_name+'-train-macro-f1', 
                    task_name+'-dev-micro-f1',  task_name+'-dev-macro-f1'])

                writer.writerow([epoch, test_lang, train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                    dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1']])
                        
                f.close()

    else:

        if(len(languages) ==1):
            task_directory = os.path.join(log_dir,'MTSL/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'MTSL/{}.csv'.format(languages[0]))
            

        else:
            task_directory = os.path.join(log_dir,'MTML/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            
            file = os.path.join(log_dir, 'MTML/log.csv')
            

        task_name_list = []

        task_f1_list = []
        #changed for task_name in task_names to for task_name in task_names:
        for task_name in task_names:
            task_name_list+=[task_name+'-train-micro-f1',  task_name+'-train-macro-f1', 
                    task_name+'-dev-micro-f1',  task_name+'-dev-macro-f1']

            task_f1_list +=[train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                    dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1']]


        if(os.path.exists(file)):
            #print("File exists: ")
            #print(file)
            #file = open(file, 'a')
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, test_lang]+  task_f1_list)

                f.close()

        else:
            #print("File does not exist: ")
            #print(file)
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['epoch', 'test_lang'] + task_name_list )
                writer.writerow([epoch, test_lang]+  task_f1_list )


                f.close()
                



#Log the final score
        
def log_score(log_dir, languages, test_lang, task_names, embeds,h_dim, cross_stitch_init,
    constraint_weight, sigma, optimizer, train_score, dev_score, test_score):
    

    if(len(task_names) ==1):
        task_name = task_names[0]

        if(len(languages) == 1):
            task_directory = os.path.join(log_dir,'STSL/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'STSL/{}_{}.csv'.format(languages[0],task_names[0]))

        else:
            task_directory = os.path.join(log_dir,'STML/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'STML/{}.csv'.format(task_names[0]))


        if(os.path.exists(file)):
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([embeds,test_lang, h_dim, cross_stitch_init, constraint_weight, sigma, optimizer,
                        train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                        dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1'], 
                        test_score[task_name]['micro_f1'], test_score[task_name]['macro_f1']])
                print([embeds,test_lang, h_dim, cross_stitch_init, constraint_weight, sigma, optimizer,
                        train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                        dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1'], 
                        test_score[task_name]['micro_f1'], test_score[task_name]['macro_f1']])
                        
        else:
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(['embeds', 'test_lang', 'h_dim', 'cross_stitch_init', 'constraint_weight', 'sigma', 'optimizer',
                       task_name+'-train-micro-f1',  task_name+'-train-macro-f1', task_name+'-dev-micro-f1',  task_name+'-dev-macro-f1', 
                       task_name+'-test-micro-f1',  task_name+'-test-macro-f1'])
                print(['embeds', 'test_lang', 'h_dim', 'cross_stitch_init', 'constraint_weight', 'sigma', 'optimizer',
                       task_name+'-train-micro-f1',  task_name+'-train-macro-f1', task_name+'-dev-micro-f1',  task_name+'-dev-macro-f1', 
                       task_name+'-test-micro-f1',  task_name+'-test-macro-f1'])

                writer.writerow([embeds,test_lang, h_dim, cross_stitch_init, constraint_weight, sigma, optimizer,\
                        train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                        dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1'], 
                        test_score[task_name]['micro_f1'], test_score[task_name]['macro_f1']])
                #added line
                #add test here
                #end of add   
                print([embeds,test_lang, h_dim, cross_stitch_init, constraint_weight, sigma, optimizer,\
                        train_score[task_name]['micro_f1'], train_score[task_name]['macro_f1'], 
                        dev_score[task_name]['micro_f1'], dev_score[task_name]['macro_f1'], 
                        test_score[task_name]['micro_f1'], test_score[task_name]['macro_f1']])


                f.close()

    else:

        if(len(languages) ==1):
            task_directory = os.path.join(log_dir,'MTSL/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'MTSL/{}.csv'.format(languages[0]))

        else:
            task_directory = os.path.join(log_dir,'MTML/')
            if not os.path.exists(task_directory):
                os.mkdir(task_directory)
            file = os.path.join(log_dir, 'MTML/log.csv')


        task_name_list = []

        task_f1_list = []
 
        for task in task_names:
            task_name_list+=[task+'-train-micro-f1', task+'-train-macro-f1', task+'-dev-micro-f1', task+'-dev-macro-f1', task+'-test-micro-f1', task+'-test-macro-f1']

            task_f1_list +=[ train_score[task]['micro_f1'], train_score[task]['macro_f1'], dev_score[task]['micro_f1'], dev_score[task]['macro_f1'], test_score[task]['micro_f1'], test_score[task]['macro_f1']]

        if(os.path.exists(file)):
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([embeds, test_lang, h_dim, cross_stitch_init, constraint_weight, sigma,optimizer]+\
                    task_f1_list)
                print([embeds, test_lang, h_dim, cross_stitch_init, constraint_weight, sigma,optimizer]+\
                    task_f1_list)


                f.close()

        else:
            with open(file, 'a') as f:
                writer = csv.writer(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['embeds', 'test_lang', 'h_dim', 'cross_stitch_init', 'constraint_weight', 'sigma']\
                    +task_name_list)
                writer.writerow([embeds, test_lang,h_dim, cross_stitch_init, constraint_weight, sigma,optimizer]+\
                    task_f1_list )
                print(['embeds', 'test_lang', 'h_dim', 'cross_stitch_init', 'constraint_weight', 'sigma']\
                    +task_name_list)
                print([embeds, test_lang,h_dim, cross_stitch_init, constraint_weight, sigma,optimizer]+\
                    task_f1_list )


                f.close()
                




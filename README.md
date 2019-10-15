
	If you use our dataset, please cite our EMNLP paper:
	@inproceedings{ousidhoum-etal-multilingual-hate-speech-2019,
    		title = "Multilingual and Multi-Aspect Hate Speech Analysis",
    		author = "Ousidhoum, Nedjma
             		and Lin, Zizheng
             		and Zhang, Hongming
            		and Song, Yangqiu
            		and Yeung, Dit-Yan",
    			booktitle = "Proceedings of EMNLP",
    		year = "2019",
    		publisher =	"Association for Computational Linguistics",
	}
	
	(You can preview our paper on https://arxiv.org/pdf/1908.11049.pdf)

Dataset

	Our dataset is composed of three csv files sorted by language. 
	They contain the tweets and the annotations described in our paper:
		the hostility type (column: tweet sentiment), 
		hostility directness (column: directness), 
		target attribute (column: target), 
		target group (column: group) and, 
		annotator's sentiment (column: annotator sentiment).

Experiments

	To replicate our experiments, please follow the guidelines below.

Requirements
	
	- Python 3.6 onwards,
	- dyNET  0.0.0 and its dependencies (follow the instructions on https://dynet.readthedocs.io/en/latest/python.html),
		[On a side note, when you install DyNet make sure that you are using CUDA 9 and CUDNN for CUDA 9. 
		 I used the following command:
	  	 CUDNN_ROOT=/path/to/conda/pkgs/cudnn-7.3.1-cuda10.0_0 \
		 BACKEND=/path/to/conda/pkgs/cudatoolkit-10.0.130-0 \
		 pip install git+https://github.com/clab/dynet#egg=dynet 
	  	 Using CUDA 10 will generate an error when calling DyNet for GPUs.
	- cross-lingual word embeddings (Babylon or MUSE. The reported results have been run using Babylon.)
		

Python files

	- annotated_data_processing.py: contains a normalization function that cleans the content of the tweets.
	- constants.py: defines constants used across all files.
	- utils.py: utility methods for data processing.
	- baseline_classifiers.py: allows you to run majority voting and logistic regression by calling:
		- run_majority_voting(train_filename, dev_filename, test_filename, attribute) or
		- run_logistic_regression(train_filename, dev_filename, test_filename, attribute)
	  on csv files of the same form of the dataset.	
	- predictors.py: Contains classes for sequence predictors and layers.
	- run_sluice_net.py: Script to train, load, and evaluate SluiceNetwork.
	- sluice_net.py: The main logic for the SluiceNetwork. (Ruder et al. 2017. 
		More details on the implementation of Sluice networks can be found on 
		https://github.com/sebastianruder/sluice-networks)	  

How to run the program

	- To save and load the trained model, you need to create a directory (e.g., model/), 
	and specify the name of the created directory when using --model-dir argument in the command line.
	- To save the log files of the training and evaluation, you need to create a directory (e.g., log/), 
	and specify the name of the created directory when using --log-dir argumnet in the command line.
	Example:
	python run_sluice_net.py --dynet-autobatch 1  --dynet-gpus 3 --dynet-seed 123 \
                          --h-layers 1 \
                         --cross-stitch\
                         --num-subspaces 2 --constraint-weight 0.1 \
                         --constrain-matrices 1 2 --patience 3 \
                         --languages ar en fr \
			 --test-languages ar en fr \
                         --model-dir model/ --log-dir log/\
                         --task-names annotator_sentiment sentiment directness group target \
			 --train-dir '/path/to/train' \
                         --dev-dir '/path/to/dev' \
                         --test-dir 'path/to/test' \
                         --embeds babylon --h-dim 200 \
			 --cross-stitch-init-scheme imbalanced \
			 --threshold 0.1
	NB: 
    	(1) The meaning of each argument can be found in run_sluice_net.py
	(2) '--task_names' refers to a list of task names (e.g: annotator_sentiment)
    	(3) '--languages' refers to the language dataset which will be used for the training. 
	(4) 'test-languages' can only be the subset of '--languages'.



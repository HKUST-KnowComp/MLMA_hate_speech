Multitask learning:
	(1) Python files:

		constants.py: Contains constants used across all files.
		predictors.py: Contains classes for sequence predictors and layers.
		run_sluice_net.py: Script to train, load, and evaluate SluiceNetwork.
		sluice_net.py: The main logic for the SluiceNetwork.
		utils.py: Utility methods for data processing.

	(2) Model directory:
		To save and load the trained model, you need to create a directory (e.g., model/) with four sub-directories inside, whose names are exactly MTML/, MTSL/, STML/ and STSL/. Then you need to specify the directory (e.g., model/) in the --model-dir argumnet in the command line.

	(3) Log directory:
		To save the log files of the training and evaluation, you need to create a directory (e.g., log/) with four sub-directories inside, whose names are exactly MTML/, MTSL/, STML/ and STSL/.Then you need to specify the directory (e.g., log/) in the --log-dir argumnet in the command line.


Example usage:


	python run_sluice_net.py --dynet-autobatch 1  --dynet-gpus 3 --dynet-seed 123 \
                          --h-layers 1 \
                         --cross-stitch\
                         --num-subspaces 2 --constraint-weight 0.1 \
                         --constrain-matrices 1 2 --patience 3 \
                         --languages ar en fr --test-languages ar en fr\
                         --model-dir model/ --log-dir log/\
                         --task-names annotator_sentiment sentiment directness group target --train-dir '/home/zlinai/MTL/data/multilingual_hate_speech/processed_data/annotated_data/train'\
                         --dev-dir '/home/zlinai/MTL/data/multilingual_hate_speech/processed_data/annotated_data/dev'\
                         --test-dir '/home/zlinai/MTL/data/multilingual_hate_speech/processed_data/annotated_data/test'\
                         --embeds babylon --h-dim 200 --cross-stitch-init-scheme imbalanced --threshold 0.1


    Note: 
    	(1) The meaning of each argument can be found in run_sluice_net.py;
    	(2) '--languages' argument refers to the language dataset whice will be used for training. 'test-languages' argument refers to the language dataset whice the model will be used for testing. So '--test-languages' can only be the subset of '--languages';



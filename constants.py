"""
Constants shared across files.
"""
import re

# special tokens and number regex
UNK = '_UNK'  # unk/OOV word/char
WORD_START = '<w>'  # word star
WORD_END = '</w>'  # word end
NUM = 'NUM'  # number normalization string
NUMBERREGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

# tasks

TASK_NAMES = ['group', #The target group of the tweet
			  'annotator_sentiment', #The sentiment of the annotator with respect to the tweet
			  'directness', #Whether the tweet is direct or indirect hate speech
			  'target', #The characteristic based on which the tweet discriminates people (e.g., race).
			  'sentiment' ] #The sentiment expressed by the tweet

# word embeddings
EMBEDS = ['babylon', 'muse', 'umwe', None]

EMBEDS_FILES = {'babylon': '../data/bi-embedding-babylon78/transformed_embeds/',
			   'muse': '../data/bi-embedding-muse/',
			   'umwe':' '}


#Dictionary of tasks and corresponding labels
LABELS = {'group':['arabs', 'other', 'african_descent', 'left_wing_people', 'asians',
 'hispanics', 'muslims', 'individual', 'special_needs', 'christian', 'immigrants', 'jews' , 
 'women', 'indian/hindu', 'gay', 'refugees'],
		'annotator_sentiment':['indifference', 'sadness', 'disgust', 'shock', 'confusion', 
		'anger', 'fear'],
		'directness':['direct', 'indirect'], 
		'target':['origin', 'religion', 'disability', 'gender', 'sexual_orientation', 'other'],
		'sentiment':['disrespectful', 'fearful', 'offensive', 'abusive', 'hateful', 'normal']}
MODIFIED_LABELS = {'group':['arabs', 'other', 'african_descent', 'left_wing_people', 'asians',
 'hispanics', 'muslims', 'individual', 'special_needs', 'christian', 'immigrants', 'jews' , 
 'women', 'indian/hindu', 'gay', 'refugees'],
		'annotator_sentiment':['indifference', 'sadness', 'shock', 'confusion','anger', 'fear'],
		'directness':['direct', 'indirect'], 
		'target':['origin', 'religion', 'disability', 'gender', 'sexual_orientation', 'other'],
		'sentiment':['somewhatoffensive', 'offensive', 'veryoffensive', 'normal']}

#'directness':['direct', 'indirect', 'none'], #to be added 

# languages
LANGUAGES = ['ar', 'en', 'fr']
FULL_LANG = {'ar': 'Arabic', 'en': 'English', 'fr': 'French'}




# optimizers
SGD = 'sgd'
ADAM = 'adam'


# cross-stitch and layer-stitch initialization schemes
BALANCED = 'balanced'
IMBALANCED = 'imbalanced'

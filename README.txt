The data sets can be obtained from: 
https://competitions.codalab.org/competitions/17344#learn_the_details-data
The ReadMes found in the downloaded files can be followed in order to obtain all the data required for the Semeval task. 

Alternatively the data and scripts can be found and accessed online via Google Colab from the files provided in the google drive directory at https://drive.google.com/drive/folders/1havlvbLokT0mjt5NsoByZ80Eie8eQO-X?usp=sharing 

!!! SAVED MODELS CAN BE FOUND IN THIS GOOGLE DRIVE
To use these saved models they need to be stored in the empty folder "models". 
 
------------------------------------------------------------------------------------------------------------------------------------

This directory contains the following files and folders: 


* models 
   contains the saved models
	* To load a saved model, open the respective model's script in jupyter notebook and run the cells marked with #+

* evaluation.py → The modified version of the official evaluation script for this task found at “https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/tools/evaluation%20script/scorer_semeval18.py” which is imported by the models


* setUpFunctions.py → includes the cleaning of tweets and retrieval of tweet dataframes (containing Id, emoji label, tweet and cleaned tweet per tweet for training, trial and test datasets)


* Bag of Words.ipynb → Jupyter notebook for Bag of Words model. The following variables can be modified in the notebook's cell. 
	vectorizerType = 1 # Count Vectorizer
	# vectorizerType = 2 # Tf-Idf Vectorizer

	language = 'en' 
	# language = 'es'


* Distilbert.ipynb → Jupyter notebook for Distilbert fine-tuning model. The following variables can be modified in the notebook's cell.
	# language = 'es'
	language = 'en'

	cleanTweetsFlag = False #Set to True to use clean tweets and False to use non-cleaned tweets


* RNN.ipynb → Jupyter notebook for RNN models. The following variables can be modified in the notebook's cell.
	language ='es'
	# language ='es'

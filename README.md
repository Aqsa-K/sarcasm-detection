# Sarcasm Detection
Sarcasm detection classifier in TensorFlow Keras

## Overview of the project:

The dataset has been taken from https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
Each record consists of three attributes:

    is_sarcastic: 1 if the record is sarcastic otherwise 0

    headline: the headline of the news article

    article_link: link to the original news article. Useful for collecting supplementary data


The model architecture is as follows:


```
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_8 (Embedding)      (None, 32, 16)            160000    
_________________________________________________________________
global_average_pooling1d_1 ( (None, 16)                0         
_________________________________________________________________
dense_12 (Dense)             (None, 24)                408       
_________________________________________________________________
dense_13 (Dense)             (None, 1)                 25        
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0 
```


Steps :
- Initially we started with a small model and only 6 neurons in the Dense layer. But that led to oveerfitting since the capacity of the model was lower than what was required. 
- Then we slowly increased the neurons in the Dense layer to 10,16,20 and 24 to improve validation accuracy
- We were using Flatten() in place of GlobalAveragePooling1D initially, so this too was a change we brought to the network
- The number of epochs were increased form an initial number of 10 to 30 since the loss was still decresing after 10, so we let the model run for more epochs

Initial hyperparameters setting:
```
vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"
training_size = 20000
```

Graphs:



Analysis :
- As can be seen from the graph, the training accuracy increases and the validation accuracy is okay
- However, if you look at the loss graph, the training loss is decreasing but the validation loss is increasing
- If you think about loss in tis context, as a confidence in the prediction. Then our result shows that while the number of accurate predictions increased over time, the confidence per prediction effectievly decreased.
- This happens a lot in text data, so it is important to keep an eye on this. The results can be improved using smaller vocab_size, smaller max_length (reducing the likelihood of padding) and rerunning. This was applied and the results were improved as can be seen from the graph in notebook



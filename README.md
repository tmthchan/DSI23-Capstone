
![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png)
# Capstone Project: Generating Kafka - Text Generation using NLP

## Executive Summary
Literature is an important part of our education, it influences us and makes us understand other walks of life. Narratives, in particular, inspire empathy and give people a new perspective on their lives and that of others. However, creating stories requires a great amount of effort to brainstorm and produce. What if we could use a language model to understand writing patterns and create new stories. This could help us generate new ideas and aid in the creative process for a new piece of work. With the text generation model, a body of text can be produced from just a few lines of input text.

Frank Kafka was a German-speaking writer renowned for developing a unique genre of stories that fuse elements of realism and fantasy. His works have influenced other famous writers such as George Orwell and Haruki Murakami. This creative style is known as kafka-esque. For this study, we will be using the complete text from the english translation of "The Metamorphosis" to build a text generation model using natural language processing.

A language model can predict the probability of the next word in the sequence, based on the words already observed in the sequence. Neural network models are a preferred method for developing statistical language models because they can use a distributed representation where different words with similar meanings have similar representation. Recurrent Neural Network is a generalization of feedforward neural network that has an internal memory. RNN is recurrent in nature as it performs the same function for every input of data while the output of the current input depends on the past one computation. Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory. The vanishing gradient problem of RNN is resolved here.

To evaluate the performance of the text generation model, we will use the cumulative BLEU scores and Rouge scores to compare our optimized models with the reference text generation model proposed by Jason Brownlee. BLEU is a precision focused metric that calculates n-gram overlap of the reference and generated texts. This n-gram overlap means the evaluation scheme is word-position independent apart from n-grams’ term associations. The ROUGE Score or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics that measures the number of matching ‘n-grams’ between the model-generated text and a ‘reference’. This indicates the extent of similarity between the generated text and the reference text sequence.

We observe that the optimized model performs better than the reference model. Looking at the difference cumulative n-gram scores, the 1-gram score for the optimized model is slightly lower than the reference model. However, the 2-gram, 3-gram and 4-gram scores are significantly higher. This shows that the generated text sequences from the optimized model have a higher similarity to the reference text. Several hyperparameters were tuned in the reference model to derive the optimized model. First, the number of hidden neurons (units) was increased from 100 to 150. This increased the accuracy of the model but did not significantly contribute to better BLEU or ROUGE scores. Subsequently, dropout layers were added after each LSTM layer to reduce overfitting. In tuning the degree of dropout, a rate of 0.2 was found to be better than 0.1 based on literature review and testing. This addition only had a slight improvement on model performance. After which, a unidirectional LSTM layer was replaced with a birectional LSTM layer to observe if sutyding the past and future inputs helped. This addition produced a significant increase in the BLEU and ROGUE scores for the optimized model.

The optimized model showed better BLEU and ROUGE scores than the reference model. Having tuned various hyperparameters in the optimized model, the two most significant changes that improved model performance were introducing a bidirectional LSTM layer and decreasing the batch size from 128 to 64. Other less significant changes include increasing the number of hidden neurons from 100 to 150, increasing the number of epochs from 100 to 150 and adding a dropout layer with dropout rate of 0.2 after each LSTM layer to reduce overfitting. Increasing the sequence length of the input text resulted in lower BLEU and ROUGE scores, hence a shorter sequence length would have better model performance. When considering different types of input text, unseen text from the same author achieved higher BLEU and ROUGE scores than an unseen text from a different author. However, the output text still referenced the main character and settings from the training text.

## 1. Introduction

### Who is Frank Kafka?

Frank Kafka was a German-speaking writer reknowned for developing a unique genre of stories that fuse elements of realism and fantasy. His works have influenced other famous writers such as George Orwell and Haruki Murakami. This creative style is known as kafka-esque.The best known works by Frank Kafka include "The Metamorphosis", "The Trial" and "The Castle". His works often feature characters facing bizarre predicaments and inflexible socio-bureaucratic powers, commonly exploring themes of alienation, existential anxiety, guilt, and absurdity.[1]

For this study, we will be using the complete text from the english translation of "The Metamorphosis" to build a text generation model using natural language processing. The complete text is available from Project Gutenberg, an online library of free ebooks to encourage the creation and distribution of ebooks.[2] To test the text generation model further, we will input a random excerpt from "The Castle" by Frank Kafka to observe how the model performs on unseen text from the same author. In addition, we will input another excerpt from "Kafka on the Shore" by Haruki Murakami to test the model on unseen text from a separate author.

### Natural Language Processing
Natural language processing is the area of study dedicated to the automatic manipulation of speech and text by software. It is an old field of study, originally dominated by rule-based methods designed by linguists, then statistical methods and, more recently, deep learning methods that show great promise in the field.[3]

A language model can predict the probability of the next word in the sequence, based on the words already observed in the sequence. Neural network models are a preferred method for developing statistical language models because they can use a distributed representation where different words with similar meanings have similar representation. Furthermore, they can use a large context of recently observed words when making predictions.

### RNN & LSTM
Recurrent Neural Network is a generalization of feedforward neural network that has an internal memory. RNN is recurrent in nature as it performs the same function for every input of data while the output of the current input depends on the past one computation. After producing the output, it is copied and sent back into the recurrent network. For making a decision, it considers the current input and the output that it has learned from the previous input.[4]

Long Short-Term Memory (LSTM) networks are a modified version of recurrent neural networks, which makes it easier to remember past data in memory. The vanishing gradient problem of RNN is resolved here. LSTM is well-suited to classify, process and predict time series given time lags of unknown duration. It trains the model by using back-propagation.

Bidirectional LSTMs can be used to train two sides, instead of one side of the input sequence. First from left to right on the input sequence and the second in reversed order of the input sequence. It provides one more context to the word to fit in the right context from words coming after and before, this results in faster and fully learning and solving a problem.[5]

## 2. Problem Statement
Literature is an important part of our education, it influences us and makes us understand other walks of life. Narratives, in particular, inspire empathy and give people a new perspective on their lives and that of others. However, creating stories requires a great amount of effort to brainstorm and produce. What if we could use a language model to understand writing patterns and create new stories. This could help us generate new ideas and aid in the creative process for a new piece of work. With the text generation model, a body of text can be produced from just a few lines of input text.

The text generation model will consider the words in the input sequence of text and predict the next possible word of the sequence. The process will be repeated until the specified number of output words are generated. The model will use natural language processing, language modelling, and deep learning. "The Metamorphosis" by Frank Kafka will be used as the reference text to create a word dictionary and generate a dataset of text sequences to train the model. The deep learning model will be built using LSTMs.

To evaluate the performance of the text generation model, we will use the cumulative BLEU scores and Rouge scores to compare our optimized models with the reference text generation model proposed by Jason Brownlee.[6]   

### BLEU Score

BLEU (Bilingual Evaluation Understudy) is a precision focused metric that calculates n-gram overlap of the reference and generated texts. This n-gram overlap means the evaluation scheme is word-position independent apart from n-grams’ term associations. One thing to note in BLEU — there is a brevity penalty i.e. a penalty applied when the generated text is too small compared to the target text. In this study, Sentence BLEU score is used to evaluate a candidate sentence against one or more reference sentences.[7]

### ROUGE Score

The ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics that measures the number of matching ‘n-grams’ between the model-generated text and a ‘reference’. This indicates the extent of similarity between the generated text and the reference text sequence.[8]

ROUGE-1 will measure the match-rate of unigrams between the model output and reference while ROUGE-2 will measure the match-rate of bigrams.  ROUGE-L measures the longest common subsequence (LCS) between our model output and reference by count the longest sequence of tokens that is shared between both. A longer shared sequence would indicate more similarity between the two sequences.

Recall counts the number of overlapping n-grams found in both the model output and reference then divides this number by the total number of n-grams in the reference. This shows how well the output text relates to the reference text. The precision metric is calculated by dividing the number of overlapping n-grams found in the output text and the reference text by the output n-gram count. This gives a score relative to the output text rather than the reference. The f1 score is calculated using the recall and precision scores to give an overall score.

## 3. Data Sources & Dictionary
|Dataset|Description|
|---|---|
|meta_text|This is the full english translated text of "The Metamorphosis" by Frank Kafka taken from Project Gutenberg.|
|castle_text|This is a small portion of text taken from The Castle by Frank Kafka.|
|kots_text|This is a small portion of text taken from Kafka on the Shore by Haruki Murakami.|
|Text_Sequences_50_meta|This is a dataset of 50 word sequences processed from "The Metamorphosis"|
|Text_Sequences_50_castle|This is a dataset of 50 word sequences processed from "The Castle"|
|Text_Sequences_50_kots|This is a dataset of 50 word sequences processed from "Kafka on the Shore"|
|Text_Sequences_100_meta|This is a dataset of 100 word sequences processed from "The Metamorphosis"|
|Text_Sequences_100_castle|This is a dataset of 100 word sequences processed from "The Castle"|
|Text_Sequences_100_kots|This is a dataset of 100 word sequences processed from "Kafka on the Shore"|

## 4. Results & Analysis

### "The Metamorphosis" Text
|        Text        |     Score Type        | Ref Model        | Ref Model LS |Opt Model  |
|:------------------:|:---------------------:|:----------------:|:------------:|:---------:|
|The Metamorphosis   |Cumulative 1-gram      |      0.2549      |    0.2871    | 0.2353    |     
|                    |Cumulative 2-gram      |      0.1428      |    0.0758    | 0.1815    |    
|                    |Cumulative 3-gram      |      0.0963      |    0.0399    | 0.1621    |    
|                    |Cumulative 4-gram      |      0.0645      |    0.0155    | 0.1432    |   

Table 1: BLEU Scores for "The Metamorphosis" Text

| Text       |Score Type | Metric | Ref Model | Ref Model LS |Opt Model |
|:------------------:|:-----------:|:-----------------:|:-------------:|:--------:|:-------------:|
|The Metamorphosis   |Rouge-1      |      Recall       |    0.2500     | 0.2740   |    0.3000     |
|                    |             |      Precision    |    0.2941     | 0.2778   |    0.2791     |
|                    |             |      F1           |    0.2703     | 0.2759   |    0.2892     |
|                    |Rouge-2      |      Recall       |    0.0800     | 0.0200   |    0.1400     |
|                    |             |      Precision    |    0.0975     | 0.0208   |    0.1429     |
|                    |             |      F1           |    0.0879     | 0.0204   |    0.1414     |
|                    |Rouge-l      |      Recall       |    0.2000     | 0.1233   |    0.2500     |
|                    |             |      Precision    |    0.2353     | 0.1250   |    0.2326     |
|                    |             |      F1           |    0.2162     | 0.1241   |    0.2410     |

Table 2: ROGUE Scores of "The Metamorphosis" Text

From the Table 1, we observe that the optimized model performs better than the reference model. Looking at the difference cumulative n-gram scores, the 1-gram score for the optimized model is slightly lower than the reference model. However, the 2-gram, 3-gram and 4-gram scores are significantly higher. This shows that the generated text sequences from the optimized model have a higher similarity to the reference text.

Several hyperparameters were tuned in the reference model to derive the optimized model. First, the number of hidden neurons (units) was increased from 100 to 150. This increased the accuracy of the model but did not significantly contribute to better BLEU or ROUGE scores. Subsequently, dropout layers were added after each LSTM layer to reduce overfitting. In tuning the degree of dropout, a rate of 0.2 was found to be better than 0.1 based on literature review and testing. This addition only had a slight improvement on model performance. After which, a unidirectional LSTM layer was replaced with a bidirectional LSTM layer to observe if sutyding the past and future inputs helped. This addition produced a significant increase in the BLEU and ROGUE scores for the optimized model.

The last two hyperparameters explored were batch size and epoch number. Increasing the number of epochs from 100 to 150 increased the accuracy of the model but did not increase the BLEU or ROGUE scores. A more impactful change was decreasing the batch size from 128 to 64, which increased the number of iterations per epoch. This significantly lowered the loss function score and improved the BLEU and ROGUE scores.

When analyzing the ROGUE scores in Table 2, the optimized model shows better scores than the reference model. Since F1-score  takes into consideration both the recall and precision of the model, it can be a good measure of model performance. The F1-score for Rouge-1 is 0.2892 for the optimized model and 0.2703 for the reference model. The F1-score for Rouge-2 is 0.1414 for the optimized model and 0.0879 for the reference model. The F1-score for Rouge-l is 0.2410 for the optimized model and 0.2162 for the reference model. This shows that the optimized model shows significant improvement than the reference model.

Besides hyperparameter tuning, the sequence length was explored to see if a longer input sequence would result in better model performance. However, it can be seen from the BLEU scores above that increasing the sequence length decreased model performance by about 50%. This was also supported by the lower ROGUE scores when a sequence length of 100 words was used. Hence, it is better to use a shorter sequence length to achieve better results.

Having explored the aforementioned hyperparameters, the two most significant changes that improved model performance were introducing a bidirectional LSTM layer and decreasing the batch size from 128 to 64.

### "The Castle" Text
| Text               |         Score Type    |  Ref Model       | Ref Model LS |Opt Model  |
|:------------------:|:---------------------:|:----------------:|:------------:|:---------:|
|The Castle          |Cumulative 1-gram      |      0.1960      |    0.2574    | 0.2352    |    
|                    |Cumulative 2-gram      |     0.0626       |    0.0507    | 0.0970    |     
|                    |Cumulative 3-gram      |     0.0208       |    0.0143    | 0.0277    |
|                    |Cumulative 4-gram      |     0.0113       |    0.0071    | 0.0141    |   

Table 3: BLEU Scores for "The Castle" Text

| Text               |Score Type   | Metric            | Ref Model     | Ref Model LS |Opt Model      |
|:------------------:|:-----------:|:-----------------:|:-------------:|:------------:|:-------------:|
|The Castle          |Rouge-1      |      Recall       |    0.1521     | 0.2615       |   0.2391      |
|                    |             |      Precision    |    0.1891     | 0.2428       |   0.2750      |
|                    |             |      F1           |    0.1686     | 0.2519       |   0.2558      |
|                    |Rouge-2      |      Recall       |    0.0200     | 0.0108       |   0.0400      |
|                    |             |      Precision    |    0.0208     | 0.0105       |   0.0408      |
|                    |             |      F1           |    0.0204     | 0.0106       |   0.0404      |
|                    |Rouge-l      |      Recall       |    0.1086     | 0.1077       |   0.2500      |
|                    |             |      Precision    |    0.1351     | 0.1000       |   0.2326      |
|                    |             |      F1           |    0.1204     | 0.1037       |   0.2410      |

Table 4: ROUGE Scores for "The Castle" Text

To observe how the text generation model performs on unseen text from the same author, we used an excerpt from "The Castle" by Frank Kafka and analyzed the BLEU and ROGUE scores. As expected, the BLEU and ROGUE scores for "The Castle" were lower than "The Metamorphosis" since the word dictionary was derived from the latter. As a result, some of the words used in "The Castle" may not be found in the model's word dictionary. This could be addressed by using GLoVe embedding with uses a global dictionary of words rather than a limited vocabulary derived from the reference text.

When comparing the performance of the optimized model to the reference model, we can see that the optimized model shows better BLEU and ROGUE scores. However, when we review the text generated after inputting the excerpt from "The Castle", the character name in the output sequence still resembles that from "The Metamorphosis", refering to the main character, Gregor. Similarly, the setting of the story references the house in "The Metamorphosis" rather than the village in "The Castle".

Perhaps multiple texts can be used to train the text generation model. This may help in the generalization of the model and help it generate multiple characters and scenarios.

### "Kafka on the Shore" Text

| Text               |      Score Type       | Ref Model      | Ref Model LS |Opt Model |
|:------------------:|:---------------------:|:--------------:|:------------:|:--------:|
|Kafka on the Shore  |Cumulative 1-gram      |    0.0392      |   0.1485     | 0.0392   |   
|                    |Cumulative 2-gram      |    0.0088      |   0.0121     | 0.0088   |
|                    |Cumulative 3-gram      |    0.0057      |   0.0056     | 0.0057   |    
|                    |Cumulative 4-gram      |    0.0043      |   0.0035     | 0.0043   |    

Table 5: BLEU Scores for "Kafka on the Shore" Text

| Text               |Score Type   | Metric            | Ref Model     | Ref Model LS |Opt Model      |
|:------------------:|:-----------:|:-----------------:|:-------------:|:------------:|:-------------:|
|Kafka on the Shore  |Rouge-1      |      Recall       |    0.0444     | 0.1232       |     0.0444    |
|                    |             |      Precision    |    0.0526     | 0.1232       |     0.0588    |
|                    |             |      F1           |    0.0482     | 0.1233       |     0.0506    |
|                    |Rouge-2      |      Recall       |    0.0000     | 0.0000       |     0.0000    |
|                    |             |      Precision    |    0.0000     | 0.0000       |     0.0000    |
|                    |             |      F1           |    0.0000     | 0.0000       |     0.0000    |
|                    |Rouge-l      |      Recall       |    0.0444     | 0.0548       |     0.0444    |
|                    |             |      Precision    |    0.0526     | 0.0548       |     0.0588    |
|                    |             |      F1           |    0.0482     | 0.0548       |     0.0506    |

Table 6: ROGUE Scores for "Kafka on the Shore" Text

Apart from text by Frank Kafka, we input an excerpt from "Kafka on the Shore" by Haruki Murakami. The purpose of this was to explore how text from another author that shares a similar style to Frank Kafka would perform when tested on the text generation model. However, we observed lower BLEU and ROGUE scores than "The Castle" text. This could be due to different gramatical structures between Frank Kafka and Haruki Murakami. Both authors created stories in different time periods, Kafka in 1920s and Murakami in 1990. As observed in model results from "The Castle", the text generated after inputing the excerpt from "Kafka on the Shore" retained the main character, Gregor, from "The Metamorphosis".

## 5. Conclusions & Recommendations

The optimized model showed better BLEU and ROUGE scores than the reference model. Having tuned various hyperparameters in the optimized model, the two most significant changes that improved model performance were introducing a bidirectional LSTM layer and decreasing the batch size from 128 to 64. Other less significant changes include increasing the number of hidden neurons from 100 to 150, increasing the number of epochs from 100 to 150 and adding a dropout layer with dropout rate of 0.2 after each LSTM layer to reduce overfitting.

Increasing the sequence length of the input text resulted in lower BLEU and ROUGE scores, hence a shorter sequence length would have better model performance. When considering different types of input text, unseen text from the same author achieved higher BLEU and ROUGE scores than an unseen text from a different author. However, the output text still referenced the main character and settings from the training text.

To improve the model, some of the following methods can be explored.
1. Use multiple books from the same author to produce a larger collection of reference text.
2. Use multiple books from several authors that have similar writing styles produced during a similar period.
3. Use GloVe word embedding vectors to increase representation of words.
4. Split the raw data based on sentences and pad each sentence to a fixed length.

## 6. References

[1] Franz Kafka - Wikipedia, accessed 7 September 2021, <https://en.wikipedia.org/wiki/Franz_Kafka> </br>
[2] Project Gutenberg, accessed 31 August 2021, <https://www.gutenberg.org/> </br>
[3] Brownlee, J. (2017), Deep Learning for Natural Language Processing: Develop Deep Learning Models for Natural Language in Python, Jason Brownlee. </br>
[4] Understanding RNN and LSTM, accessed 5 September 2021, <https://aditi-mittal.medium.com/understanding-rnn-and-lstm-f7cdf6dfc14e> </br>
[5] NLP: Text Generation through Bidirectional LSTM model, accessed 5 September 2021, <https://towardsdatascience.com/nlp-text-generation-through-bidirectional-lstm-model-9af29da4e520> </br>
[6] Brownlee, J. (2017), Deep Learning for Natural Language Processing: Develop Deep Learning Models for Natural Language in Python, Jason Brownlee, p. 224. </br>
[7] How to evaluate Text Generation Models? Metrics for Automatic Evaluation of NLP Models, accessed 1 September 2021, <https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1> </br>
[8] The Ultimate Performance Metric in NLP, accessed 1 September 2021, <https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460> </br>

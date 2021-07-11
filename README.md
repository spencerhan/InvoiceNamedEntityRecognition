# Invoice Named Entity Recognition


In this study, we adopt a supervised machine learning workflow We used a variety of Python libraries to annotate Amazon Textract OCR'ed receipt text with our chosen named entity labels and to develop three deep learning models to achieve our NER goal **BiLSTM-CRF**, **BiLSTM-CNN-CRF** and **ELMo-BiLSTM**.

The models are evaluated under three different train/validation/test split schemes We also investigated the impact of OCR noises on the result by training the model using data contain misreadings versus training the model using data with high OCR qualities.

We discovered that BiLSTM-CRF is the least expensive one to implement but produces the worst prediction result amongst the three models.
The ELMo BiLSTM with contextualised word embedding consumes the most resources to compute but yields the best prediction result amongst the three models.

Meanwhile, BiLSTM-CNN-CRF lies in the middle of the former two models With the ELMo-BiLSTM model, we retained an above 80 precision, recall and f 1 score for most named entities types except for the 'merchant' entity. 

We also presented a live example to demonstrate how the ELMo-BiLSTM model identifies named entities on an unlabelled receipt text. 

As a result, we concluded that even with the presence of OCR noises, our best performing model ELMo BiLSTM is able to produce a reliable NER result and can be reused in a production environment.

In this repository, you will find the **TensorFlow** and **spaCy** code used to implement the deep learning model together with the presentation of my findings and my final dissertation write-up. 

### A quick summary of the project can be illustrated by my offcial poster:

![Alt text](/Final_Report/SpencerHan_Poster.jpg?raw=true "Poster")

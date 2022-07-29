# MarkBERT: Marking Word Boundaries Improves Chinese BERT


*[MarkBERT: Marking Word Boundaries Improves Chinese BERT](https://arxiv.org/abs/2203.06378)*

### Fast Experiment for NER
```python
bash fastnlp-ner/run_ner.sh
```
This training example is for msra-ner. The dataset can be found in the folder msra-mark. The checkpoints can be downloaded from [markbert](https://drive.google.com/drive/folders/1RP88vvaLmPkSyjQO9Chc2-FVSw-4fYNn?usp=sharing).

You should see results like below:

ontonotes-results:



run-1

FitlogCallback evaluation on data-test:                  
span: f=0.827065, pre=0.816909, rec=0.837478               
acc_span: acc=0.978047                                     
Evaluate data in 5.5 seconds!                             
FitlogCallback evaluation on data-train:                   
span: f=0.931626, pre=0.926514, rec=0.936794               
acc_span: acc=0.990249                                    
Evaluation on dev at Epoch 3/10. Step:3762/15850:          
span: f=0.813191, pre=0.812256, rec=0.814128               
acc_span: acc=0.977757  


run-2

FitlogCallback evaluation on data-test:
span: f=0.824422, pre=0.80732, rec=0.842264
acc_span: acc=0.978673
Evaluate data in 5.48 seconds!
FitlogCallback evaluation on data-train:
span: f=0.911083, pre=0.89973, rec=0.922726
acc_span: acc=0.987973
Evaluation on dev at Epoch 2/3. Step:2970/4755:
span: f=0.806176, pre=0.797273, rec=0.81528
acc_span: acc=0.977811

run-3

FitlogCallback evaluation on data-test:                   
span: f=0.824504, pre=0.810659, rec=0.838831   
acc_span: acc=0.978354                                   
Evaluate data in 5.58 seconds!                            
FitlogCallback evaluation on data-train:       
span: f=0.938988, pre=0.932747, rec=0.945314             
acc_span: acc=0.991173                                    
Evaluation on dev at Epoch 3/5. Step:3762/7925:
span: f=0.806487, pre=0.804959, rec=0.80802              
acc_span: acc=0.978021      




msra-results

In Epoch:5/Step:21793, got best dev performance:
span: f=0.96069, pre=0.961054, rec=0.960327
acc_span: acc=0.994596



### Preprocess
We add markers in the data preprocess phase during fine-tuning therefore the usage of MarkBERT is simple.
We use *[TexSmart tookit](https://ai.tencent.com/ailab/nlp/texsmart/zh/index.html)* to do segmentation and pos-tagging in preprocessing the data.


In the CLUE experiments: You can simply use the tokenizer in run_glue.py to replace BERT tokenizers and run fine-tuning experiments in any huggingface transformers versions.
You MUST follow the dataset sample (as seen in the data_sample.txt file) to preprocess the corresponding fine-tuning dataset sothat the MarkBertTokenizer can correctly tokenize the input texts for MarkBERT. 


In the NER experiments: You also need to insert markers manually since the dataset is char-level (as seen in the data_sample.txt file), then you can use MarkBERT just like normal BERT-models.
You can use the cutoff function provided to avoid sentences over 512 tokens.

The special tokens for the markers is:

in MarkBERT, the special token is '[unused1]'.


### Usage

Without using the MarkBERT tokenizer, you can also use MarkBERT checkpoints as an improved version of BERT-BASE.

We provide a FastNLP version to quickly test the effectiveness of MarkBERT.

You can install the fastnlp and fitlog packages and enter the fastnlp folder to run the bash.

You need to prepare your train and dev file and assign the path in fastnlp-ner/run_ner.py line21-22 and assign the model checkpoint path in the fastnlp-ner/run_ner.sh 

Also, you can use MarkBERT as following the pre-process steps and then use it in huggingface Transformers or any other toolkit that operate pre-trained models.

If you encounter any errors, you may find help in https://github.com/LeeSureman/Flat-Lattice-Transformer .


### Bug Fix

We thank Hao Jiang for the help in locating an evaluation bug in the NER task in the previous MarkBERT implementation.
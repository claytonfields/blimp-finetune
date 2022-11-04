# BLiMP-FineTune

This repository provides a python script for minimal finetuning an ELECTRA model on the BLiMP task. The BLiMP task presents an NLP model with two sentences that differ by a single edit. One of the sentences is gramattically correct and the other is not. For example:

    correct sentence: "There weren't many waitresses sitting down."
    incorrect sentence: "There weren't all waitresses sitting down."

The model is assessed to be correct if it scores the correct sentence with a higer probability than the incorrect one. BLiMP was originally conceived as an unsupervised task to test NLP models without the use of fine-tuning. The ELECTRA model's novel pretraining method, however, is not readily compatible with the original evaluation strategy used in BLiMP. Therefore this program runs BLiMP with only minimal fine-tuning, using only 10 percent of the data for training, in order to apply the task as closely as possible to its original conception. Currently, this program only supports pytorch implementations of the ELECTRA transformer model. 

## Requirements

    numpy
    transformers
    torch
    sklearn
    pandas
    tqdm

## Install and Running

To install, clone repository and install dependencies.

To run from command line, navigate to the blimp-finetune directory and use :

        python3 blimp-ft <path to target directory containing ELECTRA pytorch checkpoint>

The following options are available:

        '-epochs', '-e', followed by an integer value
        '-learning_rate', '-lr', followed by a float value
        '-max_length', '-ml', followed by an integer value
        '-batch_size', '-bs', followed by an integer value


## Details

By default the program runs for 1 epoch using a learning rate 2e-5, a maximum sequence length of 128 and a batch size of 32. These parameters can all be adjusted using the options described above. Currently, the 10/90 train/test split cannot be altered. The target directory must contain an ELECTRA transformer model stored as 'pytorch_model.bin' and an appropriate 'config.json' file describing its configuration. If the model uses a vocabulary different from the standard ELECTRA model, the target directory must contain a corresponding 'tokenizer.json' file. If no 'tokenizer.json' is provided, the program will automatically download the tokenizer for the standard ELECTRA model's vocabulary from the HuggingFace transformer hub. This program will run considerably faster if a CUDA enabled GPU is available. 

## Output

Upon completion the program will store the finetuned model to an output folder along with text file, 'eval.txt' with the loss and accuracy score for each epoch of fine-tuning. Both files will be stored inside a directory whose name is the final directory in the target path provided. 

## Example

To test the program a directory containing a pretrained electra model is included with the repository. To use it simply run:

        python3 blimp-ft electra-example

This model should produce an accuracy score of around .95 using default settings.



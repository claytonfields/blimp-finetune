# BLiMP-FineTune

This repository provides a script for minimal finetuning an ELECTRA model on the BLiMP task.

## Requirements

    numpy
    transformers
    torch
    sklearn
    pandas
    tqdm

## Install and Running

    To install, clone repository and install dependencies.

    To run from command line, navigate to the blimp -finetune folder and use :

    python3 blimp-tt <path to pytorch checkpoint>

## Details

    By default runs for 1 epoch using a 10/90 train/test split

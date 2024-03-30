# prometrans

<h1 align="center">PromeTrans: Bootstrap Binary Functionality Classification with Knowledge Transferred from Pre-trained Models</h1>

<h4 align="center">
<p>
<a href=#about>About</a> |
<a href=#data>Data</a> |
<a href=#quickstart>QuickStart</a> |
<a href=#details>Details</a> |
<p>
</h4>

## About
PromeTrans is a novel framework to transfer knowledge from LLMs into assembly language model thus enhance binary program comprehension.

## Data

GPT dataset: data/GPT-dataset
POJ dataset: data/POJ-dataset

"adt_null.so@add_event": "Database module." In the GPT dataset, the add_event function in adt_null is a database module. Binaries are not yet available because of Github's storage limitation.


## QuickStart

This document will help you set up and start using the OpTrans model for embedding generation.


### Requirements
- Python 3.6 or higher
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Transformers library](https://huggingface.co/docs/transformers/installation)
- A CUDA-enabled GPU is highly recommended for faster processing.

Ensure you have Python and PyTorch installed on your system. Then, install the Transformers library using pip:
```bash
pip install transformers
```


## Details
In this document, we provide an overview of the contents of this repository and instructions for accessing the materials.


### Fine-tune
We provide a script to fine-tune the base model with your own datasets.
```bash
python Intuition/prometrans.py
```
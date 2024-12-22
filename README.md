# Torch parallel from scratch

This is a toy implementation of two data parallelization techniques commonly used: Parameter server and Ring all-reduce. For this implementation, only torch.multiprocessing module is used to handle process creation and communication. torch.multiprocessing is a wrapper for multiprocessing module in Python standard library, which enables sharing torch tensors.

## Quick start

To run the 
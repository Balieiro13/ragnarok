import os
import argparse

import pathlib


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def store_vectors_args():
    parser = argparse.ArgumentParser(
                        prog='StoreVectors',
                        description='Stores embedded vectors to ChromaDB')

    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('-c', '--collection', type=str, default="default")
    
    args = parser.parse_args()

    return args

def question_args():
    parser = argparse.ArgumentParser(
                    prog='question',
                    description='Responds a given question using Llama2 and RAG technique')

    parser.add_argument('-c', '--collection', type=str, default="default")
    parser.add_argument('-q', '--question', type=str)
    parser.add_argument('--prompt', type=argparse.FileType("r"))

    args = parser.parse_args()

    return args

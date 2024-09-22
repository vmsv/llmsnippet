import sys
from transformers import AutoTokenizer # type: ignore
from utils import *
import pprint as pp




class mytok():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("/models/Meta-Llama-3.1-8B-Instruct/")

    def tokenize(self, in_data:str) -> list:
        return self.tokenizer.tokenize(in_data)
        
if __name__ == "__main__":
    
    mtok = mytok()
    pp.pprint(f"Text: Labscan 2024 - The CTI (CyberThreat Intelligence) conferece.\n\n\nTokens:{mtok.tokenize('Labscan 2024 - The CTI (CyberThreat Intelligence) conferece.')}")
    
    mp = getMoonPeek()
    pp.pprint(f"Text: {mp}\n\n\nTokens:{mtok.tokenize(mp)}")
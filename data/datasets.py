import abc
import random
from itertools import permutations
from typing import Set
import csv
import os

class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ['o', '='] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1)*len(self.group_elements2)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs)*frac_train)], idxs[int(len(idxs)*frac_train):]
    
    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]
    
    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]
    
    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]
    
    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c]-2), equation
    
    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

class ModSumDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a + b) % self.p

class ModSubtractDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a - b) % self.p

class ModDivisonDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModDivisonDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a * pow(b, self.p-2, self.p)) % self.p

class PermutationGroup(AbstractDataset):
    def __init__(self, k, frac_train):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])

class VarBindingDataset(AbstractDataset):
    def __init__(self, csv_path, frac_train=0.5):
        self.csv_path = csv_path
        self.sequences = self._load_sequences()

        # Extract all variables and values from the sequences
        all_vars = set()
        all_values = set()

        for seq in self.sequences:
            # Parse the sequence to extract variables and values
            assignments = seq.split(',')
            for assignment in assignments:
                if '=' in assignment:
                    var, val = assignment.split('=')
                    all_vars.add(var)
                    all_values.add(val)

        # Initialize parent class with dummy group elements (not used for this dataset)
        super(VarBindingDataset, self).__init__(set(), set(), frac_train)

        # Override the vocabulary and other attributes set by parent class
        self.idx2vocab = ['='] + sorted(list(all_vars)) + sorted(list(all_values))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(all_values)



        # Override the pairs with our sequence indices
        idxs = list(range(len(self.sequences)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs)*frac_train)], idxs[int(len(idxs)*frac_train):]

    def _load_sequences(self):
        """Load variable binding sequences from CSV file."""
        sequences = []

        with open(self.csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    sequences.append(line)

        return sequences

    def _parse_sequence(self, sequence_str):
        """Parse a sequence string into tokens."""
        assignments = sequence_str.split(',')
        tokens = []
        for assignment in assignments:
            if '=' in assignment:
                var, val = assignment.split('=')
                tokens.extend([var, '=', val])
        return tokens

    def fetch_output(self, a, b):
        # This method is required by AbstractDataset but not used in our implementation
        return None

    def fetch_example(self, idx):
        sequence_str = self.sequences[idx]
        tokens = self._parse_sequence(sequence_str)

        # Use all tokens except the last value as input, last value as target
        input_tokens = tokens[:-1]
        target_token = tokens[-1]

        # Encode the tokens
        encoded_input = [self.vocab2idx[token] for token in input_tokens]
        target_idx = self.vocab2idx[target_token]

        return encoded_input, target_idx, tokens

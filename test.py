import torch
n_letters = 2
vocab=['(', ')']
labels = ['valid', 'invalid']

def encode_sentence(sentence):

    rep = torch.zeros(len(sentence),1,n_letters)



    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos]=1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    return torch.tensor([labels.index(label)], dtype=torch.float32)
print(encode_sentence('(())'))

print(encode_labels('invalid'))
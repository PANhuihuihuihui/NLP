#some config
train = "train.pt"
test = "test.pt"
vocab_dir = "vocab.pt"

# form SAMA .pt data to text file
train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)
print("train: ",train_dataset)
print("test: ",test_dataset)
print("vocb: ",gVocab)

gold_result = []






from data.dataloader import get_data_and_vocab

cleaned_vocab, destination = get_data_and_vocab("../data/train.csv", "cleaned.csv", ["Comment", "Insult"])

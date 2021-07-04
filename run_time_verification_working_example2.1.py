
"""test my own dataset"""

import random
import torch
import pandas as pd
import numpy as np
import sys
from torchtext.legacy import data
import torch.nn as nn
from pathlib import Path

epochs_num = sys.argv[1]
samples_num = sys.argv[2]
file_index = sys.argv[3]
data_folder = sys.argv[4]
logs_folder = sys.argv[5]

Path("{}".format(str(logs_folder))).mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(columns=["accuracy_rnn", "accuracy_prob", "train_len", "train_size", "train_difficult", "new_data_test"])

with open(f'{logs_folder}/{file_index}.txt', 'w') as f:

    vocab = [char for char in 'abcd']

    TEXT = data.Field()
    LABEL = data.Field(dtype=torch.float)

    fields = [('text', TEXT), ('label', LABEL)]

    df_test = pd.read_csv(f'{data_folder}/test{file_index}.csv')
    df_train = pd.read_csv(f'{data_folder}/train{file_index}.csv')
    df_valid = pd.read_csv(f'{data_folder}/valid{file_index}.csv')

    filtered_df = df_test.loc[(df_test['label'] > 0) & (df_test['label'] < 1)]
    prec = (len(filtered_df) / len(df_test)) * 100

    filtered_df.to_csv(f"{data_folder}/test_clean{file_index}.csv", index=False, header=True)

    correct = 0
    for i in range(len(df_test)):
        # correct += (1 if np.abs(df_test.iloc[i]['label'] - df_valid.iloc[i]['label']) < 0.1 else 0)
        correct += (1 - np.abs(df_test.iloc[i]['label'] - df_valid.iloc[i]['label']))

    longest_path = df_train['text'].str.len().max()
    longest_path_test = df_test['text'].str.len().max()

    print("accuracy no rnn ", (correct / len(df_test))*100, file=f)
    print("len", longest_path_test/longest_path, file=f)
    print("difficult data", prec, file=f)
    print("train size", len(df_train), file=f)

    df.at[0, 'train_difficult'] = prec
    df.at[0, 'accuracy_prob'] = (correct / len(df_test)) * 100
    df.at[0, 'train_len'] = longest_path_test/longest_path
    df.at[0, 'train_size'] = len(df_train)

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = f'{data_folder}',
                                            train = f'train{file_index}.csv',
                                            validation = f'valid{file_index}.csv',
                                            test = f'test_clean{file_index}.csv',
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = True
    )

    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    MAX_VOCAB_SIZE = 4

    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)

    BATCH_SIZE = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        sort_key=lambda x:len(x.text),
        sort_within_batch=False,
        device = device)

    class RNN(nn.Module):
        def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

            super().__init__()

            self.embedding = nn.Embedding(input_dim, embedding_dim)

            self.rnn = nn.RNN(embedding_dim, hidden_dim)

            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, text):

            #text = [sent len, batch size]
            # print(text)
            embedded = self.embedding(text)

            #embedded = [sent len, batch size, emb dim]

            output, hidden = self.rnn(embedded)

            # output = torch.sigmoid(output)
            #output = [sent len, batch size, hid dim]
            #hidden = [1, batch size, hid dim]

            assert torch.equal(output[-1,:,:], hidden.squeeze(0))

            # return self.fc(hidden.squeeze(0))
            out = self.fc(hidden)
            # print(out)
            return torch.sigmoid(out)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 25
    HIDDEN_DIM = 50
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    import torch.optim as optim

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        # rounded_preds = torch.round(torch.sigmoid(preds))

        # print(torch.sigmoid(preds),torch.sigmoid(y) )
        # acc = 1 - np.abs((preds - y).detach()).numpy()
        # acc = 1 - np.abs((preds - y).detach().cpu()).numpy()
        acc = 1 - torch.abs(preds - y)
        return acc

        # if np.abs(torch.sigmoid(preds) - torch.sigmoid(y)) < \
        #   np.abs(0.15 * torch.sigmoid(y)):
        #   n_correct += 1
        # else:
        #   n_wrong += 1
        # return (n_correct * 100.0) / (n_correct + n_wrong)

    def train(model, iterator, optimizer, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:

            optimizer.zero_grad()

            predictions = model(batch.text).squeeze(1)
            # predictions = torch.transpose(predictions, 0, 1) #rf
            # loss = criterion(predictions, batch.label)
            loss = criterion(predictions, torch.sigmoid(batch.label))

            acc = binary_accuracy(predictions, torch.sigmoid(batch.label))

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():

            for batch in iterator:

                predictions = model(batch.text).squeeze(1)

                # print(batch.text, predictions, batch.label)
                # predictions = torch.transpose(predictions, 0, 1) #rf
                # print(predictions)
                loss = criterion(predictions, torch.sigmoid(batch.label))

                acc = binary_accuracy(predictions, torch.sigmoid(batch.label))

                print(predictions, torch.sigmoid(batch.label))

                # print(torch.sigmoid(batch.label)[0].item())
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    import time

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = int(epochs_num)

    best_valid_loss = float('inf')

    n_seq_los = 0
    last_train_acc = 0
    last_valid_acc = 0
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss and train_acc >= last_train_acc and valid_acc >= last_valid_acc:
        if valid_loss < best_valid_loss:
            n_seq_los = 0
            last_train_acc = train_acc
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '{}/ut1-model-{}-{}.pt'.format(logs_folder ,file_index, samples_num))
        else:
            n_seq_los+=1

        if n_seq_los > 5:
            # pass
          break

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('{}/ut1-model-{}-{}.pt'.format(logs_folder ,file_index, samples_num)))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    test_acc_prob = 100*(test_acc*len(filtered_df) + 1*(len(df_test) - len(filtered_df)))/len(df_test)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc_prob:.2f}%', file=f)
    df.at[0, 'accuracy_rnn'] = test_acc_prob


    # create a new column with the count of how many times the row exists
    df_train['count'] = 0
    df_test['count'] = 0
    df_test['count'] = df_test.groupby(df_test.columns.to_list()[:-1]).cumcount() + 1
    df_train['count'] = df_train.groupby(df_train.columns.to_list()[:-1]).cumcount() + 1

    # merge the two data frames with and outer join, add an indicator variable
    # to show where each row (including the count) exists.
    df_all = df_test.merge(df_train, on=df_test.columns.to_list(), how='outer', indicator='exists')
    # print(df_all)

    # clean up exists column and export the rows do not exist in both frames
    df_all['exists'] = (df_all.exists.str.replace('left_only', 'file1')
                                     .str.replace('right_only', 'file2'))
    different_df = df_all.query('exists != "both"')#.to_csv('{data_folder}/update.csv', index=False)
    print("train is different in " ,(len(different_df)/(len(df_test)+len(df_train)))*100, "%", file=f)
    df.at[0, 'new_data_test'] = (len(different_df)/(len(df_test)+len(df_train)))*100
    df.to_csv(f'{logs_folder}/{file_index}.csv', index=False)
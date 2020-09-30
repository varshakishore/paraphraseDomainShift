from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from utils import loadData
import argparse
import os
import logging


tasks = ['msr', 'quora', 'twitter', 'paws', 'paws_qqp']

logging.basicConfig(format = '%(asctime)s -%(levelname)s- %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(x, tokenizer, max_len):
    # Given two sentences, x["string1"] and x["string2"], this function returns BERT ready inputs.
    inputs = tokenizer.encode_plus(
            x["utt1"],
            x["utt2"],
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
            )

    # First `input_ids` is a sequence of id-type representation of input string.
    # Second `token_type_ids` is sequence identifier to show model the span of "string1" and "string2" individually.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = max_len - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Super simple validation.
    assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
    assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(len(attention_mask), max_len)
    assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(len(token_type_ids), max_len)

    # Convert them into PyTorch format.
    label = torch.tensor(int(x["paraphrase"])).long()
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    return {
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
            }

def getDataloaders(data_path, task, tokenizer, max_len, batch_size, evalModel=False):
    
    train, test, val = loadData(data_path, task)
    
    if evalModel:
        test_data = test.apply(preprocess, axis=1, args=[tokenizer, max_len])
        
        test_dataloader = DataLoader(
            list(test_data),
            sampler=SequentialSampler(list(test_data)),
            batch_size=args.batch_size
            )
        return test_dataloader


    train_data = train.apply(preprocess, axis=1, args=[tokenizer, max_len])
    val_data = val.apply(preprocess, axis=1, args=[tokenizer, max_len])
    test_data = test.apply(preprocess, axis=1, args=[tokenizer, max_len])

    train_dataloader = DataLoader(
                train_data,
                sampler=RandomSampler(list(train_data)),
                batch_size=batch_size
                )
    val_dataloader = DataLoader(
                val_data,
                sampler=SequentialSampler(list(val_data)),
                batch_size=batch_size
                )
    test_dataloader = DataLoader(
            list(test_data),
            sampler=SequentialSampler(list(test_data)),
            batch_size=batch_size
            )
    return train_dataloader, test_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--num_labels', default=2, type=int, help='number of labels is 2 for paraphrase detection')
    parser.add_argument('--max_len', default=256, type=int, help='max length of sentence for bert encoding')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--save_path', default='/home/vk352/paraphraseDomainShift/savedModels/', type=str, help='save directory')
    parser.add_argument('--data_path', default='/home/vk352/paraphraseDomainShift/data/', type=str, help='data directory')
    parser.add_argument('--task', default=tasks[0], type=str, choices=tasks, help='tasks in {}'.format(tasks))
    parser.add_argument('--train_task', default=tasks[0], type=str, choices=tasks, help='tasks in {}'.format(tasks))
    parser.add_argument('--eval_model', action='store_true', help='only evaluate')
    
    args = parser.parse_args()
    
    logger.info('args: {}'.format(args))
    logger.info('Task name: %s ' % (args.task))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger.info("Setting up model")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels, return_dict=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_dataloader, test_dataloader, val_dataloader = getDataloaders(args.data_path, args.task, tokenizer, args.max_len, args.batch_size)
    epochs = args.epochs
    best_dev_acc = 0
    
    # setup optimizer
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    logger.info('Num training samples: %d ' % (len(train_dataloader)*args.batch_size))
    model.to(device)

    for epoch_i in range(0, epochs):
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')

        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        for step, batch in enumerate(train_dataloader):
            if step%1000==0:
                logger.info('%d completed epochs, %d batches' % (epoch_i, step))
            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            model.zero_grad()        

            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(input_ids, 
                                 token_type_ids=token_type_ids, 
                                 attention_mask=attention_mask, 
                                 labels=labels)[:2]
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).flatten()
            total_train_accuracy += ((preds == labels).cpu().numpy().mean() * 100)
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)

        logger.info("Training accuracy: {0:.2f}".format(avg_train_accuracy))
        logger.info("Training loss: {0:.2f}".format(avg_train_loss))


        logger.info('Validation...')
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:
            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.no_grad():        

                loss, logits = model(input_ids, 
                                 token_type_ids=token_type_ids, 
                                 attention_mask=attention_mask, 
                                 labels=labels)[:2]

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            total_eval_accuracy += ((preds == labels).cpu().numpy().mean() * 100)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        logger.info(" Val Accuracy: {0:.2f}".format(avg_val_accuracy))

        if avg_val_accuracy >= best_dev_acc:
                torch.save(model.state_dict(), args.save_path+'bert_'+args.task+'.pt')
                best_dev_acc = avg_val_accuracy

    if args.eval_model:
        test_dataloader = getDataloaders(args.data_path, args.task, tokenizer, args.max_len, args.batch_size, evalModel=evalModel)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels, return_dict=True)
        model.load_state_dict(torch.load(args.save_path+'bert_'+args.train_task+'.pt'))

        model.to(device)

        # test
        model.eval()
        total_test_accuracy = 0
        total_test_loss = 0
        for batch in test_dataloader:

            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.no_grad():        

                loss, logits = model(input_ids, 
                                 token_type_ids=token_type_ids, 
                                 attention_mask=attention_mask, 
                                 labels=labels)[:2]

            # Accumulate the validation loss.
            total_test_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            total_test_accuracy += ((preds == labels).cpu().numpy().mean() * 100)


        # Report the final accuracy for this run.
        avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        avg_test_loss = total_test_loss / len(test_dataloader)
        logger.info(" Test Accuracy: {0:.2f}".format(avg_test_accuracy))

if __name__ == "__main__":
    main()
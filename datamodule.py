from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
class UITQADataModule(LightningDataModule):

    def __init__(
        self,
        model_checkpoint: str,
        task_name: str = 'uit_qa',
        max_seq_length: int = 364,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_checkpoint)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
    
    def setup(self, stage: str):
        train_df = pd.read_json(path_or_buf='data/squad_mrc_train.jsonl', lines=True)
        eval_df = pd.read_json(path_or_buf='data/squad_mrc_dev.jsonl', lines=True)
        train_ds = Dataset.from_pandas(train_df).shuffle()
        eval_ds = Dataset.from_pandas(eval_df)
    
        train_ds = train_ds.map(
            self.preprocess_train_data,
            batched = True,
            batch_size= 50,
            num_proc= 2,
            remove_columns=train_ds.column_names,)
        self.train_ds = train_ds.filter(lambda x: x['valid'],num_proc= 2).remove_columns("valid")
       
        eval_ds = eval_ds.map(
            self.preprocess_train_data,
            batched = True,
            batch_size= 50,
            num_proc= 2,
            remove_columns = eval_ds.column_names,)
        self.eval_ds = eval_ds.filter(lambda x: x['valid'],num_proc= 2).remove_columns("valid")

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.hparams.model_checkpoint, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.hparams.batch_size, collate_fn= self.data_collator, shuffle= True, num_workers =10)

    def val_dataloader(self):
        return DataLoader(self.eval_ds, batch_size = self.hparams.batch_size, collate_fn = self.data_collator, num_workers =10)

    def test_dataloader(self):
        pass

    def preprocess_train_data(self, batch):
        inputs = self.tokenizer(
            batch['question'],
            batch['context'],
            max_length=self.hparams.max_seq_length,
            truncation = True,
            return_offsets_mapping= True,
        )

        offset_mapping = inputs.pop('offset_mapping')
        start_positions = []
        end_positions = []
        valid = []
        answer_text_batch = batch['answer_text']
        answer_start_idx_batch = batch['answer_start_idx']
        for i,offset in enumerate(offset_mapping):
            answer = answer_text_batch[i]
            start_char = answer_start_idx_batch[i]
            end_char = start_char + len(answer)
            sequence_ids  = inputs.sequence_ids(i)
            idx = 0 
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx-1
            if answer == '':
                start_positions.append(0)
                end_positions.append(0)
                valid.append(True)
            elif offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
                valid.append(False)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] < start_char:
                    idx +=1
                start_positions.append(idx-1)

                while idx >= context_start and offset[idx][1] < end_char:
                    idx += 1
                
                end_positions.append(idx)
                valid.append(True)

            # Double check
            # input_ids = inputs['input_ids'][i]
            # print('answer: ', answer)
            # print('double check: ', start_positions[-1],end_positions[-1], tokenizer.decode(input_ids[start_positions[-1]:end_positions[-1]+1]))
            # print('valid', valid[-1])

        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions
        inputs['valid'] = valid
        return inputs
    
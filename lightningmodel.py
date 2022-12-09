from transformers import (
    AutoConfig, 
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import datasets
from pytorch_lightning import LightningModule 
from datetime import datetime
from typing import Optional
from evaluate import load
import torch
import numpy as np

class UITQATransformer(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UITQAModel")
        # parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument('--model_checkpoint', type = str, default = 'binhquoc/vie-deberta-small')
        parser.add_argument("--data_path", type=str, default="data/")
        return parent_parser

    def __init__(
        self,
        model_checkpoint: str,
        task_name: str = 'uit_qa',
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        n_best: int = 20,
        max_length_answer: int = 50,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.metric = load('squad_v2')
    
    def forward(self, **inputs):
        return self.model(
            input_ids = inputs['input_ids'], 
            token_type_ids = inputs['token_type_ids'], 
            attention_mask =inputs['attention_mask'],
            start_positions = inputs['start_positions'],
            end_positions = inputs['end_positions']
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss',loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, start_logits, end_logits = outputs[0:3]
        return {"loss": val_loss, 'batch': batch, 'start_logits': start_logits, 'end_logits':end_logits}

    def validation_epoch_end(self, validation_step_outputs):
        predictions = []
        example_id = 0
        references = []    
        for batch in validation_step_outputs:
            batch_size = batch['start_logits'].size()[0]
            for i in range(batch_size):
                start_logit = batch['start_logits'][i].detach().cpu().numpy()
                end_logit = batch['end_logits'][i].detach().cpu().numpy()
                input_ids = batch['batch']['input_ids'][i].detach().cpu().numpy()
                token_type_ids = batch['batch']['token_type_ids'][i].detach().cpu().numpy()
                start_position = batch['batch']['start_positions'][i].item()
                end_position = batch['batch']['end_positions'][i].item()
                if start_position == 0 and end_position == 0:
                    answer = ''
                else:
                    answer = self.tokenizer.decode(input_ids[start_position:end_position+1])
                # print(start_position, end_position, answer)
                start_indexes = np.argsort(start_logit)[-self.hparams.n_best - 1 :][::-1].tolist()
                end_indexes = np.argsort(end_logit)[-self.hparams.n_best - 1 :][::-1].tolist()

                cand_pred_answers= []
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if token_type_ids[start_index] == 0 or token_type_ids[end_index] == 0:
                            continue
                        if end_index < start_index or end_index - start_index + 1 < self.hparams.max_length_answer:
                            continue
                        cand_answer = {
                            "text": self.tokenizer.decode(input_ids[start_index:end_index]),
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        cand_pred_answers.append(cand_answer)
                if len(cand_pred_answers) > 0:
                    best_answer = max(cand_pred_answers, key=lambda x: x["logit_score"])
                    predictions.append(
                        {"id": str(example_id), "prediction_text": best_answer["text"], 'no_answer_probability': 1-best_answer['logit_score']}
                    )
                else:
                    predictions.append({"id": str(example_id), "prediction_text": "", 'no_answer_probability': 1.0})
                if answer:
                    references.append({"id": str(example_id), "answers": {'answer_start':[start_position],'text': [answer]}})
                else:
                    references.append({"id": str(example_id), "answers": {'answer_start':[],'text': []}})
                example_id += 1  

        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=predictions, references=references, no_answer_threshold= 0.2), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay"""
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams.learning_rate, eps = self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
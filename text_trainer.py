
from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np


from additional_loss import BPRloss

class TextTrainer(Trainer):
    def __init__(self, args_original, **kwds):
        
        super().__init__(**kwds)
        self.args_original = args_original

        self.loss_fn_list = [('bce', nn.BCEWithLogitsLoss(), 1.0)]        
        if args_original.use_bpr_loss:
            self.loss_fn_list.append( ('bpr', BPRloss(), args_original.bpr_loss_weight) )
            print("Using BPR loss in addition to BCEWithLogitsLoss")
        
        
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        
        if self.args_original.model_name == 'AvsHModel':

            total_label = inputs.pop("total_labels", None)
            total_label = total_label.float() if total_label is not None else None
            paragraph_label = inputs.pop("paragraph_labels", None)
            paragraph_label = paragraph_label.float() if paragraph_label is not None else None

            total_logits, paragraph_logits = model(**inputs)

            if paragraph_label is None:
                paragraph_label = torch.zeros_like(paragraph_logits, dtype=torch.float, device=paragraph_logits.device)
            
            total_loss = torch.tensor(0.0, device=total_logits.device)
            if not self.args_original.split_valid_by_paragraph:
                for loss_name, loss_fn, weight in self.loss_fn_list:
                    
                    if loss_name=='bce':
                        #process paragraph_label
                        flat_paragraph_logits = paragraph_logits.view(-1)
                        flat_paragraph_label = paragraph_label.view(-1)
                        flat_total_logits = total_logits.view(-1)
                        flat_total_label = total_label.view(-1)
                        
                        flat_paragraph_logits = flat_paragraph_logits[flat_paragraph_label != -1]  # remove padding
                        flat_paragraph_label = flat_paragraph_label[flat_paragraph_label != -1]  # remove padding

                        total_loss += loss_fn(flat_total_logits, flat_total_label) * weight
                        total_loss += loss_fn(flat_paragraph_logits, flat_paragraph_label) * weight
                    else:
                        total_loss += loss_fn(total_logits.view(-1), total_label.view(-1)) * weight
                        total_loss += loss_fn(paragraph_logits, paragraph_label) * weight
                        
            label = total_label
            logits = total_logits.view(-1)


        else:
            label = inputs.pop("labels", None)
            label = label.float() if label is not None else None
            
            output = model(**inputs)
            logits = output.logits.view(-1)

            if label is None: #dummy label for evaluation
                label = torch.zeros_like(logits, dtype=torch.float, device=logits.device)

            total_loss = torch.tensor(0.0, device=logits.device)
            if not self.args_original.split_valid_by_paragraph:
                for _, loss_fn, weight in self.loss_fn_list:
                    loss = loss_fn(logits, label) * weight
                    total_loss += loss


        if return_outputs:
            return total_loss, logits, label
        return total_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            *args, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        
        if self.args_original.split_valid_by_paragraph:
            pred = pred.view(1,-1)
            label = label.view(1,-1)
        
        return (eval_loss,pred,label)


from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl

class StopAfterEpoch(TrainerCallback):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        # state.epoch는 현재 epoch (float) 값을 가집니다.
        if state.epoch >= self.max_epochs:
            control.should_training_stop = True
        return control

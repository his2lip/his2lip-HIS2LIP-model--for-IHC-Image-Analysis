import pdb

import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

from tqdm import tqdm



class Evaluator:
    
    def __init__(self,
        medclip_clf,
        eval_dataloader=None,
        mode=None,
        ) -> None:
        
        self.clf = medclip_clf
        self.mode = mode
        self.eval_dataloader = eval_dataloader
    
    def evaluate(self, eval_dataloader=None):
        self.clf.eval()
        if self.eval_dataloader is None and eval_dataloader is not None: self.eval_dataloader = eval_dataloader
        else: eval_dataloader = self.eval_dataloader
        pred_list = []
        label_list = []
        for data in tqdm(eval_dataloader, desc='Evaluation'):
            with torch.no_grad():
                outputs = self.clf(**data)
                print(f"Logits Shape: {outputs['logits'].shape}") 
                pred = outputs['logits']
            pred_list.append(pred)
            label_list.append(data['labels'])
        
        pred_list = torch.cat(pred_list, 0)
        labels = torch.cat(label_list, 0).cpu().detach().numpy()

        pred = pred_list.cpu().detach().numpy()        
        outputs = {'pred':pred, 'labels':labels}

        if self.mode is None:
            if len(labels.shape) == 1:
                if len(np.unique(labels)) == 2:
                    self.mode = 'binary'
                else:
                    self.mode = 'multiclass'
            else:
                self.mode = 'multilabel'
            print(f'no mode specified, will pick mode `{self.mode}` by data.')

        if self.mode == 'binary':
            if pred.shape[1] == 1:
                pred_score = torch.tensor(pred).sigmoid().numpy().flatten()
                auc = roc_auc_score(labels, pred_score)
                outputs['auc'] = auc
                pred_label = np.ones(len(pred))
                pred_label[pred_score<0.5] = 0
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc
                

            else: # have 2 outputs
                pred_score = torch.tensor(pred).sigmoid().numpy()
                pred_label = np.argmax(pred_score, 1)
                acc = (pred_label == labels).mean()
                outputs['acc'] = acc


            res = classification_report(labels, pred_label, output_dict=True)
            res = res['macro avg']
            res.pop('support')
            outputs.update(res)

        if self.mode == 'multiclass':
            # Ensure predictions are valid
            if pred.ndim != 2:
                raise ValueError(f"Expected 2D predictions, got shape {pred.shape}")

            # Ensure labels are 1D
            if len(labels.shape) > 1:
                labels = labels.squeeze()

            pred_label = pred.argmax(1)  # Get the class with the highest score
            print(f"Pred Shape: {pred.shape}, Predicted Labels Shape: {pred_label.shape}, Labels Shape: {labels.shape}")

            # Validate shapes
            if pred_label.shape != labels.shape:
                raise ValueError(f"Shape mismatch: Predicted labels shape {pred_label.shape} and labels shape {labels.shape}")

            # Compute accuracy
            acc = (pred_label == labels).mean()
            outputs['acc'] = acc

            # Compute classification report
            res = classification_report(labels, pred_label, output_dict=True, zero_division=0)
            res = res['macro avg']
            res.pop('support')  # Remove support metric
            outputs.update(res)
            print(f"Accuracy: {acc}, Classification Report: {res}")

        return outputs
    
    def process_confusion_matrix(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        outputs = {}
        # Sensitivity, hit rate, recall, or true positive rate
        outputs['tpr'] = TP/(TP+FN)
        # Specificity or true negative rate
        outputs['tnr'] = TN/(TN+FP) 
        # Precision or positive predictive value
        outputs['ppv'] = TP/(TP+FP)
        # Negative predictive value
        outputs['npv'] = TN/(TN+FN)
        # Fall out or false positive rate
        outputs['fpr'] = FP/(FP+TN)
        # False negative rate
        outputs['fnr'] = FN/(TP+FN)
        # False discovery rate
        outputs['fdr'] = FP/(TP+FP)

        # Overall accuracy for each class
        # outputs['acc'] = (TP+TN)/(TP+FP+FN+TN)
        if cnf_matrix.shape[0] > 2: # multiclass
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = np.mean(v)
        else:
            for k,v in outputs.items(): # take macro avg over each class
                outputs[k] = v[1]
        return outputs
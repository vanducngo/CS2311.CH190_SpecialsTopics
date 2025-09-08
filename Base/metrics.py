import torch
from sklearn.metrics import roc_auc_score
import numpy as np

class AUCProcessor:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes

        if class_names is not None and len(class_names) == num_classes:
            self.class_names = class_names
        else:
            if class_names is not None:
                print(f"Warning: `class_names` length ({len(class_names)}) does not match `num_classes` ({num_classes}). Using default names.")
            self.class_names = [f"Class {i}" for i in range(num_classes)]

        self.predictions_by_domain = {}
        self.labels_by_domain = {}

        self.all_predictions = []
        self.all_labels = []
        self.all_domains = []
        self._calculated = False

    def process(self, predictions, labels, domains):
        self._calculated = False
        
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self.all_predictions.append(predictions_np)
        self.all_labels.append(labels_np)
        
        if isinstance(domains, torch.Tensor):
            domains = [str(d.item()) for d in domains]
        self.all_domains.extend(domains)

        
        # Tung domain
        for i in range(len(domains)):
            domain = domains[i]
            if domain not in self.predictions_by_domain:
                self.predictions_by_domain[domain] = []
                self.labels_by_domain[domain] = []
            
            self.predictions_by_domain[domain].append(predictions_np[i])
            self.labels_by_domain[domain].append(labels_np[i])

    def calculate(self):
        self.results_per_domain = {}
        all_preds_concat = []
        all_labels_concat = []

        # Lặp qua từng domain đã thu thập
        for domain in self.predictions_by_domain.keys():
            preds = np.array(self.predictions_by_domain[domain])
            labels = np.array(self.labels_by_domain[domain])
            
            all_preds_concat.append(preds)
            all_labels_concat.append(labels)

            per_class_auc = {}
            valid_aucs = []
            for i in range(self.num_classes):
                if len(np.unique(labels[:, i])) > 1:
                    auc = roc_auc_score(labels[:, i], preds[:, i])
                    per_class_auc[f"auc/{self.class_names[i]}"] = auc
                    valid_aucs.append(auc)
                else:
                    per_class_auc[f"auc/{self.class_names[i]}"] = float('nan')
            
            mean_auc = np.nanmean(valid_aucs) if valid_aucs else 0.0
            self.results_per_domain[domain] = {'mean_auc': mean_auc, 'per_class_auc': per_class_auc}
        
        if all_preds_concat:
            overall_preds = np.concatenate(all_preds_concat, axis=0)
            overall_labels = np.concatenate(all_labels_concat, axis=0)
            
            overall_per_class_auc = {}
            overall_valid_aucs = []
            
            for i in range(self.num_classes):
                class_name = self.class_names[i]
                # Kiểm tra trên toàn bộ dữ liệu gộp
                if len(np.unique(overall_labels[:, i])) > 1:
                    auc = roc_auc_score(overall_labels[:, i], overall_preds[:, i])
                    overall_per_class_auc[class_name] = auc
                    overall_valid_aucs.append(auc)
                else:
                    overall_per_class_auc[class_name] = float('nan')
            
            overall_mean_auc = np.nanmean(overall_valid_aucs) if overall_valid_aucs else 0.0
            
            self.overall_results = {
                'mean_auc': overall_mean_auc,
                'per_class_auc': overall_per_class_auc
            }

    def info(self):
        output_str = ""
        overall_aucs = []
        for domain, results in self.results_per_domain.items():
            output_str += f"\n--- Results for Domain: {domain} ---\n"
            output_str += f"- Mean AUC: {results['mean_auc']:.4f}\n"
            overall_aucs.append(results['mean_auc'])
            for class_name, auc in results['per_class_auc'].items():
                output_str += f"- {class_name}: {auc:.4f}\n"

        # Tính trung bình của các Mean AUC
        if overall_aucs:
            final_mean = np.mean(overall_aucs)
            output_str += f"\n==================================================\n"
            output_str += f"OVERALL MEAN AUC (avg of per-corruption AUCs): {final_mean:.4f}\n"
            output_str += f"=================================================="
        return output_str
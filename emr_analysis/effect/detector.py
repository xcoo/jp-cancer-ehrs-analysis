from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset

import emr_analysis
from emr_analysis.data import Effect
from emr_analysis.effect.dataset import ReportDataset
from emr_analysis.tokenizer import StanzaTokenizer, simple_tokenize
from emr_analysis.treatment_period import TreatmentPeriod


sys.modules['prism_emr_analyzer'] = emr_analysis


class LSTMClassifier0(nn.Module):
    def __init__(self, max_reports, max_tokens, embedding_dim, hidden_dim, bidirectional=False,
                 num_layers=1, dropout=0):
        super(LSTMClassifier0, self).__init__()

        self.max_reports = max_reports
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, embeddings):
        xs = []
        for x in embeddings:
            out, (hn, _) = self.lstm(x)
            xs.append(hn[0])
        return torch.stack(xs)

    def save(self, f) -> None:
        obj = {"max_reports": self.max_reports,
               "max_tokens": self.max_tokens,
               "embedding_dim": self.embedding_dim,
               "hidden_dim": self.hidden_dim,
               "bidirectional": self.bidirectional,
               "num_layers": self.num_layers,
               "dropout": self.dropout,
               "state_dict": self.state_dict()}
        torch.save(obj, f)

    @staticmethod
    def load(f, device) -> LSTMClassifier0:
        obj = torch.load(f, map_location=device, weights_only=False)
        model = LSTMClassifier0(obj["max_reports"], obj["max_tokens"], obj["embedding_dim"],
                                obj["hidden_dim"], obj["bidirectional"],
                                num_layers=obj["num_layers"], dropout=obj["dropout"])
        model.load_state_dict(obj["state_dict"])
        return model


class LSTMClassifier(nn.Module):
    def __init__(self, max_reports, max_tokens, int2effect_dict, embedding_dim, hidden_dim,
                 label_size, bidirectional=False, num_layers=1, dropout=0):
        super(LSTMClassifier, self).__init__()

        self.max_reports = max_reports
        self.max_tokens = max_tokens
        self.int2effect_dict = int2effect_dict
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        hidden_dim_ = hidden_dim
        if bidirectional:
            hidden_dim_ *= 2
        self.hidden2tag = nn.Linear(hidden_dim_ * num_layers, label_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden2pd = nn.Linear(hidden_dim_, 3)
        self.softmax_pd = nn.LogSoftmax(dim=2)

    def forward(self, embeddings):
        out, (hn, _) = self.lstm(embeddings)

        tag_space = self.hidden2tag(hn[0])
        tag_scores = self.softmax(tag_space)

        pd_space = self.hidden2pd(out)
        pd_scores = self.softmax_pd(pd_space)

        return tag_scores, pd_scores

    def save(self, f) -> None:
        obj = {"max_reports": self.max_reports,
               "max_tokens": self.max_tokens,
               "int2effect_dict": self.int2effect_dict,
               "embedding_dim": self.embedding_dim,
               "hidden_dim": self.hidden_dim,
               "label_size": self.label_size,
               "bidirectional": self.bidirectional,
               "num_layers": self.num_layers,
               "dropout": self.dropout,
               "state_dict": self.state_dict()}
        torch.save(obj, f)

    @staticmethod
    def load(f, device) -> LSTMClassifier:
        obj = torch.load(f, map_location=device, weights_only=False)
        model = LSTMClassifier(obj["max_reports"], obj["max_tokens"], obj["int2effect_dict"],
                               obj["embedding_dim"], obj["hidden_dim"], obj["label_size"],
                               obj["bidirectional"], num_layers=obj["num_layers"],
                               dropout=obj["dropout"])
        model.load_state_dict(obj["state_dict"])
        return model


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Trainer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _calc_weight(self, xs, label_size, ignore_index=None):
        size = len([x for x in xs if x != ignore_index])
        weight = []
        for label in range(label_size):
            if label == ignore_index:
                w = 1.0
            else:
                n = len([x for x in xs if x == label])
                w = 1 - n / size
            weight.append(w)
        return torch.tensor(weight)

    def _optimizer(self, model, lr, weight_decay):
        return optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _scheduler(self, optimizer, gamma):
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _process_batch(self, model, x_batch, y0_batch, y1_batch, criterion0=None, criterion1=None):
        x_tensor = x_batch.to(self.device)
        y0_tensor = y0_batch.to(self.device)
        y1_tensor = y1_batch.to(self.device)

        out0, out1 = model(x_tensor)

        batch_loss0 = None
        batch_loss1 = None
        if criterion0:
            batch_loss0 = criterion0(out0, y0_tensor)
        if criterion1:
            batch_loss1 = 0
            for i, x in enumerate(out1):
                batch_loss1 += criterion1(x, y1_tensor[i])
            batch_loss1 /= len(out1)

        preds0 = []
        trues0 = []
        acc0 = 0
        acc0_sum = 0
        _, predicts = torch.max(out0, 1)
        for i, ans in enumerate(y0_tensor):
            preds0.append(predicts[i].item())
            trues0.append(ans.item())
            if predicts[i].item() == ans.item():
                acc0 += 1
            acc0_sum += 1

        preds1 = []
        trues1 = []
        acc1 = 0
        acc1_sum = 0
        _, predicts = torch.max(out1, 2)
        for i, ans in enumerate(y1_tensor):
            for j, ans2 in enumerate(ans):
                if ans2.item() == 0:
                    continue
                preds1.append(predicts[i][j].item())
                trues1.append(ans2.item())
                if predicts[i][j].item() == ans2.item():
                    acc1 += 1
                acc1_sum += 1

        return batch_loss0, batch_loss1, \
            (acc0, acc0_sum), (acc1, acc1_sum), \
            (preds0, trues0), (preds1, trues1)

    def _save_model(self, layer0, layer1, output, epoch):
        output = Path(output)
        tmpdir = output.with_name(output.stem + ".tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        tmpout = tmpdir / (output.stem + f".{epoch}" + output.suffix)

        layer0.save(tmpout.with_suffix(".0.pt"))
        layer1.save(tmpout.with_suffix(".1.pt"))

    def train(self, dataset, test_dataset, output="model.pt", batch_size=128, epoch=100,
              summary_writer=None, hidden_sizes=(32, 16), loss_weights=(1.0, 1.0),
              loss0_label_weights=1.0, loss1_label_weights=1.0,
              lr=0.0001, weight_decay=0.0001, lr_gamma=0.985, dropout=0.2):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                 num_workers=2, pin_memory=True)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2,
                                      pin_memory=True)

        if isinstance(dataset, Subset):
            max_reports = dataset.dataset.report_size
            max_tokens = dataset.dataset.token_size
            embedding_dim = dataset.dataset.word_embedding.vector_size
        else:
            max_reports = dataset.report_size
            max_tokens = dataset.token_size
            embedding_dim = dataset.word_embedding.vector_size

        label_size = dataset.effect_size

        layer0 = LSTMClassifier0(max_reports, max_tokens, embedding_dim, hidden_sizes[0], False)
        layer1 = LSTMClassifier(max_reports, max_tokens, dataset.int2effect_dict, hidden_sizes[0],
                                hidden_sizes[1], label_size, False)
        model = Sequential(layer0, nn.Dropout(p=dropout), layer1).to(self.device)

        if type(loss0_label_weights) is int:
            loss0_label_weights = [loss0_label_weights] * label_size
        y0_weight = (self._calc_weight([v[1] for v in DataLoader(dataset)], label_size) *
                     torch.tensor(loss0_label_weights)).to(self.device)
        loss_function = nn.NLLLoss(weight=y0_weight)
        print(f"loss weight 0: {y0_weight.tolist()}")

        if type(loss1_label_weights) is int:
            loss1_label_weights = [loss1_label_weights] * 3
        y1_flt = sum([v[2].tolist()[0] for v in DataLoader(dataset)], [])
        y1_weight = (self._calc_weight(y1_flt, 3, ignore_index=None) *
                     torch.tensor(loss1_label_weights)).to(self.device)
        loss_function2 = nn.NLLLoss(weight=y1_weight, ignore_index=-1)
        print(f"loss weight 1: {y1_weight.tolist()}")

        optimizer = self._optimizer(model, lr, weight_decay)
        scheduler = self._scheduler(optimizer, lr_gamma)

        print("Training model...")
        for epoch_i in range(epoch):
            model.train()
            losses = []
            accs0 = []
            accs1 = []
            for x_batch, y0_batch, y1_batch in data_loader:
                batch_loss0, batch_loss1, batch_acc0, batch_acc1, _, _ = \
                    self._process_batch(model, x_batch, y0_batch, y1_batch, loss_function,
                                        loss_function2)
                batch_loss = batch_loss0 * loss_weights[0] + batch_loss1 * loss_weights[1]

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                losses.append((batch_loss.item(), batch_loss0.item(), batch_loss1.item()))
                accs0.append(batch_acc0)
                accs1.append(batch_acc1)
            self._save_model(layer0, layer1, output, epoch_i)
            scheduler.step()

            train_loss = np.mean(losses, axis=0)
            acc0x, acc0y = np.sum(accs0, axis=0)
            train_acc0 = acc0x / acc0y
            acc1x, acc1y = np.sum(accs1, axis=0)
            train_acc1 = acc1x / acc1y
            print(f"epoch {epoch_i}: loss {train_loss[0]} "
                  f"({train_loss[1]}, {train_loss[2]}), "
                  f"acc {train_acc0} {train_acc1}")

            if summary_writer:
                model.eval()
                losses = []
                accs0 = []
                accs1 = []
                with torch.no_grad():
                    for x_batch, y0_batch, y1_batch in test_data_loader:
                        batch_loss0, batch_loss1, batch_acc0, batch_acc1, _, _ = \
                            self._process_batch(model, x_batch, y0_batch, y1_batch, loss_function,
                                                loss_function2)
                        batch_loss = batch_loss0 * loss_weights[0] + batch_loss1 * loss_weights[1]

                        losses.append((batch_loss.item(), batch_loss0.item(), batch_loss1.item()))
                        accs0.append(batch_acc0)
                        accs1.append(batch_acc1)
                test_loss = np.mean(losses, axis=0)
                acc0x, acc0y = np.sum(accs0, axis=0)
                test_acc0 = acc0x / acc0y
                acc1x, acc1y = np.sum(accs1, axis=0)
                test_acc1 = acc1x / acc1y

                summary_writer.add_scalars("loss/all",
                                           {"train": train_loss[0], "test": test_loss[0]},
                                           epoch_i)
                summary_writer.add_scalars("loss/effect",
                                           {"train": train_loss[1], "test": test_loss[1]},
                                           epoch_i)
                summary_writer.add_scalars("loss/pd_date",
                                           {"train": train_loss[2], "test": test_loss[2]},
                                           epoch_i)

                summary_writer.add_scalars("accuracy/effect",
                                           {"train": train_acc0, "test": test_acc0},
                                           epoch_i)
                summary_writer.add_scalars("accuracy/pd_date",
                                           {"train": train_acc1, "test": test_acc1},
                                           epoch_i)

        print("Model trained")

        print("\nModel's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        layer0.save(Path(output).with_suffix(".0.pt"))
        layer1.save(Path(output).with_suffix(".1.pt"))
        print("\nModel saved:", output)

    def test(self, dataset, model="model.pt", batch_size=128):
        layer0 = LSTMClassifier0.load(Path(model).with_suffix(".0.pt"), self.device)
        layer1 = LSTMClassifier.load(Path(model).with_suffix(".1.pt"), self.device)
        model = Sequential(layer0, nn.Dropout(), layer1).to(self.device)
        model.eval()

        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        accs0 = []
        accs1 = []
        preds0 = []
        trues0 = []
        preds1 = []
        trues1 = []
        for x_batch, y0_batch, y1_batch in data_loader:
            _, _, batch_acc0, batch_acc1, batch_pt0, batch_pt1 = \
                self._process_batch(model, x_batch, y0_batch, y1_batch)
            accs0.append(batch_acc0)
            accs1.append(batch_acc1)
            preds0 += batch_pt0[0]
            trues0 += batch_pt0[1]
            preds1 += batch_pt1[0]
            trues1 += batch_pt1[1]
        acc0x, acc0y = np.sum(accs0, axis=0)
        acc0 = acc0x / acc0y
        acc1x, acc1y = np.sum(accs1, axis=0)
        acc1 = acc1x / acc1y

        print("\nPredict (effect):", acc0)
        print("Predict (pd_date):", acc1)

        print(metrics.classification_report(trues0, preds0, labels=range(dataset.effect_size),
                                            target_names=[e.name for e
                                                          in dataset.int2effect_dict.values()]))
        print(metrics.classification_report(trues1, preds1, labels=[1, 2],
                                            target_names=["True", "False"]))


class EffectDetector(object):
    def __init__(self, data, config, model_file):
        self.word_embedding = data.word_embedding

        self.tokenizer = StanzaTokenizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layer0 = LSTMClassifier0.load(Path(model_file).with_suffix(".0.pt"), self.device)
        self.layer1 = LSTMClassifier.load(Path(model_file).with_suffix(".1.pt"), self.device)
        self.model = Sequential(self.layer0, nn.Dropout(), self.layer1).to(self.device)
        self.model.eval()
        self.layer0.lstm.train()
        self.layer1.lstm.train()

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.word_embedding.vectors))

        self.treatment_period = TreatmentPeriod(data, config)

        self.emr_radiation_report = data.emr_radiation_report

        def forward(x):
            out, _ = self.model(x)
            return out

    def detect_effect(self, patient_id, start_dt, end_dt) \
            -> tuple[Optional[Effect], Optional[list[bool]], dict[str, Any]]:
        orig_reports = ReportDataset.fetch_reports(self.emr_radiation_report, patient_id, start_dt,
                                                   end_dt)

        if orig_reports.empty:
            return None, None, {"reports": [], "pds": [], "pd_values": []}

        reports = ReportDataset.thinout_reports(orig_reports, self.layer0.max_reports)
        texts = [ReportDataset.report_to_text(report) for _, report in reports.iterrows()]
        if len(texts) < self.layer0.max_reports:
            texts = [""] * (self.layer0.max_reports - len(texts)) + texts

        tokenss = [simple_tokenize(self.tokenizer, text) for text in texts]
        tokenss = [ReportDataset.fill_padding(tokens, self.layer0.max_tokens)
                   for tokens in tokenss]
        tokens_indices = []
        for tokens in tokenss:
            unk_index = self.word_embedding.get_index("<unk>")
            tokens_indices.append([self.word_embedding.get_index(t, unk_index)
                                   for t in tokens])

        x_tensor = self.embedding(torch.tensor([tokens_indices])).to(self.device)
        out, pd_out = self.model(x_tensor)
        _, predict = torch.max(out, 1)
        values2, indices2 = torch.max(pd_out, 2)
        values2 = values2.tolist()[0]
        indices2 = indices2.tolist()[0]
        if len(reports) < self.layer0.max_reports:
            values2 = values2[-len(reports):]
            indices2 = indices2[-len(reports):]
        pd_dates = [(r, v) for r, v, i in zip(reports["ExaminDateTime"], values2, indices2)
                    if i == 1]
        pd_date = None
        if len(pd_dates) > 0:
            pd_date = max(pd_dates, key=lambda x: x[1])[0]

        return self.layer1.int2effect_dict[predict.item()], pd_date, len(reports)

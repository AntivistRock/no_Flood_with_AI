import pandas as pd
import numpy as np

import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
import tqdm

from datetime import datetime, timedelta

import traceback
from functools import partial
from tqdm.contrib.concurrent import process_map
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from tqdm.notebook import tqdm
import pickle
import pathlib
import argparse

torch.manual_seed(420)
np.random.seed(420)
torch.cuda.manual_seed(420)

import warnings
warnings.filterwarnings('ignore')


class StoreItemDataset(Dataset):
    def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=True,
                 ohe_cat_columns=False):
        super().__init__()
        self.sequence_data = None
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.cat_classes = {}
        self.cat_embed_shape = []
        self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
        self.pass_decoder_input = decoder_input
        self.ohe_cat_columns = ohe_cat_columns
        self.cat_columns_to_decoder = False

    def get_embedding_shape(self):
        return self.cat_embed_shape

    def load_sequence_data(self, processed_data):
        self.sequence_data = processed_data

    # кодирует категориальные фичи, создает шаблон кодирования категориальных фичей
    def process_cat_columns(self, column_map=None):
        column_map = column_map if column_map is not None else {}
        for col in self.cat_columns:
            self.sequence_data[col] = self.sequence_data[col].astype('category')
            if col in column_map:
                self.sequence_data[col] = self.sequence_data[col].cat.set_categories(column_map[col]).fillna('#NA#')
            else:
                self.sequence_data[col].cat.add_categories('#NA#', inplace=True)
            self.cat_embed_shape.append(
                (len(self.sequence_data[col].cat.categories), self.cat_embed_vector_size.get(col, 50)))

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        row = self.sequence_data.iloc[[idx]]
        x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
        y = torch.tensor(row['y_sequence'].values[0], dtype=torch.float32)
        if self.pass_decoder_input:
            decoder_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
        if len(self.num_columns) > 0:
            for col in self.num_columns:
                num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
                x_inputs[0] = torch.cat((x_inputs[0], num_tensor.repeat(x_inputs[0].size(0)).unsqueeze(1)), axis=1)
                decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)),
                                          axis=1)
        if len(self.cat_columns) > 0:
            if self.ohe_cat_columns:
                for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
                    col_tensor = torch.zeros(num_classes, dtype=torch.float32)
                    col_tensor[row[self.cat_columns[ci]].cat.codes.values[0]] = 1.0
                    col_tensor_x = col_tensor.repeat(x_inputs[0].size(0), 1)
                    x_inputs[0] = torch.cat((x_inputs[0], col_tensor_x), axis=1)
                    if self.pass_decoder_input and self.cat_columns_to_decoder:
                        col_tensor_y = col_tensor.repeat(decoder_input.size(0), 1)
                        decoder_input = torch.cat((decoder_input, col_tensor_y), axis=1)
            else:
                cat_tensor = torch.tensor(
                    [row[col].cat.codes.values[0] for col in self.cat_columns],
                    dtype=torch.long
                )
                x_inputs.append(cat_tensor)
        if self.pass_decoder_input:
            x_inputs.append(decoder_input)
            y = torch.tensor(row['y_sequence'].values[0][:, 0], dtype=torch.float32)
        if len(x_inputs) > 1:
            return tuple(x_inputs), y
        return x_inputs[0], y


# энкодер
class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False,
                 device='cpu', rnn_dropout=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )
        self.device = device

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden


# декодер
class DecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, dropout=0.2):
        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.attention = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_hidden, y):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, self.dropout(rnn_hidden)


# общая модель
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder_cell, output_size=3, teacher_forcing=0.3, sequence_len=336, decoder_input=True,
                 device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

    def forward(self, xb, yb=None):
        if self.decoder_input:
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return outputs


# класс для тренировки нейросети
class TorchTrainer():
    def __init__(self, name, model, optimizer, loss_fn, scheduler, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.name = name
        self.checkpoint_path = pathlib.Path(kwargs.get('checkpoint_folder', f'./models/{name}_chkpts'))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.train_checkpoint_interval = kwargs.get('train_checkpoint_interval', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 25)
        self.writer = SummaryWriter(f'runs/{name}')
        self.scheduler_batch_step = kwargs.get('scheduler_batch_step', False)
        self.additional_metric_fns = kwargs.get('additional_metric_fns', {})
        self.additional_metric_fns = self.additional_metric_fns.items()
        self.pass_y = kwargs.get('pass_y', False)
        self.valid_losses = {}

    def _get_checkpoints(self, name=None):
        checkpoints = []
        checkpoint_path = self.checkpoint_path if name is None else pathlib.Path(f'./models/{name}_chkpts')
        for cp in self.checkpoint_path.glob('checkpoint_*'):
            checkpoint_name = str(cp).split('/')[-1]
            checkpoint_epoch = int(checkpoint_name.split('_')[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        return checkpoints

    def _clean_outdated_checkpoints(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints:]:
                delete_cp[0].unlink()
                print(f'removed checkpoint of epoch - {delete_cp[1]}')

    def _save_checkpoint(self, epoch, valid_loss=None):
        self._clean_outdated_checkpoints()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': [o.state_dict() for o in self.optimizer] if type(
                self.optimizer) is list else self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint.update({
                'scheduler_state_dict': [o.state_dict() for o in self.scheduler] if type(
                    self.scheduler) is list else self.scheduler.state_dict()
            })
        if valid_loss:
            checkpoint.update({'loss': valid_loss})
        torch.save(checkpoint, self.checkpoint_path / f'checkpoint_{epoch}')
        save_dict(self.checkpoint_path, 'valid_losses', self.valid_losses)
        print(f'saved checkpoint for epoch {epoch}')
        self._clean_outdated_checkpoints()

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        if name is None:
            checkpoints = self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[0]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_model:
                if type(self.optimizer) is list:
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(checkpoint['optimizer_state_dict'][i])
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None:
                    if type(self.scheduler) is list:
                        for i in range(len(self.scheduler)):
                            self.scheduler[i].load_state_dict(checkpoint['scheduler_state_dict'][i])
                    else:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint['epoch']
        return None

    def _load_best_checkpoint(self):
        if self.valid_losses:
            best_epoch = sorted(self.valid_losses.items(), key=lambda x: x[1])[0][0]
            loaded_epoch = self._load_checkpoint(epoch=best_epoch, only_model=True)

    def _step_optim(self):
        if type(self.optimizer) is list:
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _step_scheduler(self, valid_loss=None):
        if type(self.scheduler) is list:
            for i in range(len(self.scheduler)):
                if self.scheduler[i].__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler[i].step(valid_loss)
                else:
                    self.scheduler[i].step()
        else:
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

    def _loss_batch(self, xb, yb, optimize, pass_y, additional_metrics=None):
        if type(xb) is list:
            xb = [xbi.to(self.device) for xbi in xb]
        else:
            xb = xb.to(self.device)
        yb = yb.to(self.device)
        if pass_y:
            y_pred = self.model(xb, yb)
        else:
            y_pred = self.model(xb)
        loss = self.loss_fn(y_pred, yb)
        if additional_metrics is not None:
            additional_metrics = [fn(y_pred, yb) for name, fn in additional_metrics]
        if optimize:
            loss.backward()
            self._step_optim()
        loss_value = loss.item()
        del xb
        del yb
        del y_pred
        del loss
        if additional_metrics is not None:
            return loss_value, additional_metrics
        return loss_value

    def evaluate(self, dataloader):
        self.model.eval()
        eval_bar = tqdm(dataloader, leave=False)
        with torch.no_grad():
            loss_values = [self._loss_batch(xb, yb, False, False, self.additional_metric_fns) for xb, yb in eval_bar]
            if len(loss_values[0]) > 1:
                loss_value = np.mean([lv[0] for lv in loss_values])
                additional_metrics = np.mean([lv[1] for lv in loss_values], axis=0)
                additional_metrics_result = {name: result for (name, fn), result in
                                             zip(self.additional_metric_fns, additional_metrics)}
                return loss_value, additional_metrics_result
            else:
                loss_value = np.mean(loss_values)
                return loss_value, None

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                if type(xb) is list:
                    xb = [xbi.to(self.device) for xbi in xb]
                else:
                    xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_pred = self.model(xb)
                predictions.append(y_pred.cpu().numpy())
        return np.concatenate(predictions)

    def predict_one(self, x):
        self.model.eval()
        with torch.no_grad():
            if type(x) is list:
                x = [xi.to(self.device).unsqueeze(0) for xi in x]
            else:
                x = x.to(self.device).unsqueeze(0)
            y_pred = self.model(x)
            if self.device == 'cuda':
                y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            return y_pred

    def lr_find(self, dl, optimizer, start_lr=1e-7, end_lr=1e-2, num_iter=200):
        lr_finder = LRFinder(self.model, optimizer, self.loss_fn, device=self.device)
        lr_finder.range_test(dl, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)

    def train(self, epochs, train_dataloader, valid_dataloader=None, resume=True, resume_only_model=False):
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i in tqdm(range(start_epoch, start_epoch + epochs), leave=True):
            self.model.train()
            training_losses = []
            running_loss = 0
            training_bar = tqdm(train_dataloader, leave=False)
            for it, (xb, yb) in enumerate(training_bar):
                loss = self._loss_batch(xb, yb, True, self.pass_y)
                running_loss += loss
                training_bar.set_description("loss %.4f" % loss)
                if it % 100 == 99:
                    self.writer.add_scalar('training loss', running_loss / 100, i * len(train_dataloader) + it)
                    training_losses.append(running_loss / 100)
                    running_loss = 0
                if self.scheduler is not None and self.scheduler_batch_step:
                    self._step_scheduler()
            print(f'Training loss at epoch {i + 1} - {np.mean(training_losses)}')
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar('validation loss', valid_loss, i)
                if additional_metrics is not None:
                    print(additional_metrics)
                print(f'Valid loss at epoch {i + 1} - {valid_loss}')
                self.valid_losses[i + 1] = valid_loss
            if self.scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i + 1)


def ret_val(x):
    if pd.isna(x[idx]):
        if x['date'] > pd.to_datetime('1985-01-01'):
            val_prev = daily_new.loc[x['date'] - pd.offsets.DateOffset(years=1)][idx]
            if not np.isnan(val_prev):
                return val_prev
        if x['date'] < pd.to_datetime(f'{BEGIN.year-1}-12-31'):
            val_post = daily_new.loc[x['date'] + pd.offsets.DateOffset(years=1)][idx]
            if not np.isnan(val_post):
                return val_post
    return x[idx]


# Вспомогательная функция для соединения идентичных датафреймов
def weary_append(x, y):
    if x is not None:
        if any([a for a in list(x.columns) if a not in y.columns]) or any(
                [a for a in list(y.columns) if a not in x.columns]):
            raise ValueError()
        else:
            return x.append(y)
    else:
        return y


# считает расстояние от станции до всех других
def find_good(df, lat, lon):
    shir = 111.134861111
    dolg = 111.321377778  # экваториальная
    return (((df['lat'] - lat) * shir) ** 2 + ((df['lon'] - lon) * dolg * np.cos(lon)) ** 2) ** 0.5


# для метеостанции находит три ближайших к ней
def make_variants(st_cord, post, i):
    length = find_good(st_cord.drop([post]), st_cord.loc[post].lat, st_cord.loc[post].lon)
    if i == 3 or length.min() > 100: print(f'{i} values are counted.'), print(arr); return 0
    post_new = length.idxmin()
    arr.append(post_new)
    make_variants(st_cord.drop([post]), post_new, i + 1)


def sin_transform(values):
    return np.sin(2 * np.pi * values / len(set(values)))


def cos_transform(values):
    return np.cos(2 * np.pi * values / len(set(values)))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def split_sequence_difference(group_data, n_steps_in, n_steps_out, x_cols, y_col, diff, additional_columns):
    try:
        X, y = list(), list()
        additional_col_map = defaultdict(list)
        group_data[y_col] = group_data[y_col].diff()
        additional_col_map['x_base'] = []
        additional_col_map['y_base'] = []
        additional_col_map['mean_traffic'] = []
        for i in range(diff, len(group_data)):
            # find the end of this pattern
            x_base = group_data.iloc[i - 1]['unmod_y']
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(group_data) - 1:
                break
            y_base = group_data.iloc[end_ix - 1]['unmod_y']
            # gather input and output parts of the pattern
            if len(x_cols) == 1:
                x_cols = x_cols[0]
            seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][
                y_col].values
            for col in additional_columns:
                additional_col_map[col].append(group_data.iloc[end_ix][col])
            additional_col_map['x_base'].append(x_base)
            additional_col_map['y_base'].append(y_base)
            additional_col_map['mean_traffic'] = group_data['unmod_y'].mean()
            X.append(seq_x)
            y.append(seq_y)
        additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
        return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])
    except Exception as e:
        print(e)
        print(group_data.shape)
        traceback.print_exc()


# split a multivariate sequence into samples
def split_sequences(group_data, n_steps_in, n_steps_out, x_cols, y_cols, additional_columns, step=1, lag_fns=[]):
    X, y = list(), list()
    additional_col_map = defaultdict(list)
    group_data = group_data.sort_values('date')
    for i, lag_fn in enumerate(lag_fns):
        group_data[f'lag_{i}'] = lag_fn(group_data[y_cols[0]])
    steps = list(range(0, len(group_data), step))
    if step != 1 and steps[-1] != (len(group_data) - 1):
        steps.append((len(group_data) - 1))
    for i in steps:
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(group_data):
            break
        # gather input and output parts of the pattern
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][
            y_cols + [f'lag_{i}' for i in range(len(lag_fns))]].values
        for col in additional_columns:
            additional_col_map[col].append(group_data.iloc[end_ix][col])
        X.append(seq_x)
        y.append(seq_y)
    additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
    return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])


def _apply_df(args):
    df, func, key_column = args
    result = df.groupby(key_column).progress_apply(func)
    return result


def almost_equal_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def mp_apply(df, func, key_column):
    workers = 6
    key_splits = almost_equal_split(df[key_column].unique(), workers)
    split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]
    result = process_map(_apply_df, [(d, func, key_column) for d in split_dfs], max_workers=workers)
    return pd.concat(result)


def sequence_builder(data, n_steps_in, n_steps_out, key_column, x_cols, y_col, y_cols, additional_columns, diff=False,
                     lag_fns=[], step=1):
    if diff:
        sequence_fn = partial(
            split_sequence_difference,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_col=y_col,
            diff=diff,
            additional_columns=list(set([key_column] + additional_columns))
        )
        data['unmod_y'] = data[y_col]
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + [y_col, 'unmod_y'] + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    else:
        # first entry in y_cols should be the target variable
        sequence_fn = partial(
            split_sequences,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_cols=y_cols,
            additional_columns=list(set([key_column] + additional_columns)),
            lag_fns=lag_fns,
            step=step
        )
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    sequence_data = pd.DataFrame(sequence_data, columns=['result'])
    s = sequence_data.apply(lambda x: pd.Series(zip(*[col for col in x['result']])), axis=1).stack().reset_index(
        level=1, drop=True)
    s.name = 'result'
    sequence_data = sequence_data.drop('result', axis=1).join(s)
    sequence_data['result'] = pd.Series(sequence_data['result'])
    if diff:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(
            set([key_column] + additional_columns + ['x_base', 'y_base', 'mean_traffic']))] = pd.DataFrame(
            sequence_data.result.values.tolist(), index=sequence_data.index)
    else:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns))] = pd.DataFrame(
            sequence_data.result.values.tolist(), index=sequence_data.index)
    sequence_data.drop('result', axis=1, inplace=True)
    if key_column in sequence_data.columns:
        sequence_data.drop(key_column, axis=1, inplace=True)
    sequence_data = sequence_data.reset_index()
    print(sequence_data.shape)
    sequence_data = sequence_data[~sequence_data['x_sequence'].isnull()]
    return sequence_data


def last_year_lag(col): return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)


# задаем NSE метрику
def NSE(predictions, actual):
    predictions = predictions.to_numpy()
    actual = actual.to_numpy()
    return 1 - np.sum((predictions - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)


# функция для сохранения весов сети
def save_dict(path, name, _dict):
    with open(path/f'{name}.pickle', 'wb') as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def rescale_data(scale_map, data_df, columns=['predictions', 'y_sequence', 'x_sequence']):
    rescaled_data = pd.DataFrame()
    for station_id, item_data in tqdm(data_df.groupby('station_id', as_index=False)):
        if station_id == '#NA#': continue
        mu = scale_map[station_id]['mu']
        sigma = scale_map[station_id]['sigma']
        for col in columns:
            item_data[col] = item_data[col].apply(lambda x: (np.array(x) * sigma) + mu)
        rescaled_data = pd.concat([rescaled_data, item_data], ignore_index=True)
    return rescaled_data


def generate_flat_df(sequence_data, predict_col='predictions', actual_col='Y'):
    flat_df = pd.DataFrame()
    for i, row in sequence_data.iterrows():
        row_df = pd.DataFrame()
        start_date = row['date']
        row_df['date'] = pd.date_range(BEGIN, periods=out_len).date.tolist()
        row_df['station_id'] = row['station_id']
        row_df['predictions'] = row[predict_col]
        if actual_col:
            row_df['Y'] = row[actual_col]
        flat_df = pd.concat([flat_df, row_df], ignore_index=False)
    flat_df.index = range(len(flat_df))
    flat_df['date'] = pd.to_datetime(flat_df['date'])
    return flat_df



def meteo_melt(df, col):
    return pd.melt(df.reset_index(), var_name='station_id', value_name=col, id_vars=['date'],
                   value_vars=[6005, 6022, 6027, 5004, 5012, 5024, 5805])


parser = argparse.ArgumentParser(description='Preprocessing script')
parser.add_argument('-p', action="store", dest="path_to_data", required=True)
parser.add_argument('-l', action="store", default=80,  dest="seq_len", type=int)
parser.add_argument(action="store", dest="BEGIN", type=pd.Timestamp)
parser.add_argument(action="store", dest="END", type=pd.Timestamp)
args = parser.parse_args()

path_to_data = args.path_to_data
seq_len = args.seq_len
BEGIN = args.BEGIN
END = args.END

out_len = (END - BEGIN).days

# импортируем данные
daily = pd.read_pickle(path_to_data + 'processed_data/daily.pkl')
disch_d = pd.read_pickle(path_to_data + 'processed_data/disch_d.pkl')
disch_m = pd.read_pickle(path_to_data + 'processed_data/disch_m.pkl')
st_cord = pd.read_pickle(path_to_data + 'processed_data/station_coords.pkl')
print('Hydro features are uploaded. Preparing...')
# Сортируем датасеты по номеру станции и дате замера
daily = daily.sort_values(by=['station_id', 'date'])
daily = daily.reset_index(drop=True)

disch_d = disch_d.sort_values(by=['station_id', 'date'])
disch_d = disch_d.reset_index(drop=True)

disch_m['date'] = pd.to_datetime(
    disch_m['year'].astype('str') + disch_m['month'].astype('str'), format='%Y%m'
)  # создадим даты из колонок "year" и "month" в датасете disch_m

disch_m = disch_m.sort_values(by=['station_id', 'date'])
disch_m = disch_m.reset_index(drop=True)

st_cord = st_cord.set_index('station_id')
st_cord = st_cord.sort_values(by=['station_id'])

# Выберем в датасетах только целевые посты
station_ids = [6005, 6022, 6027, 5004, 5012, 5024, 5805]
daily_purp = daily[daily['station_id'].isin(station_ids)]
disch_d_purp = disch_d[disch_d['station_id'].isin(station_ids)]
disch_m_purp = disch_m[disch_m['station_id'].isin(station_ids)]

# delete outliers from hydro post number 5012 (will fill by interpolation)
daily_purp.loc[(daily_purp['stage_max'] > daily_purp['stage_max'].quantile(0.99)) \
&(daily_purp['station_id'] == 5012), 'stage_max'] = np.nan
print('Hydro features are prepared.')

# Соединим датасеты daily и disch_d с сохранением всех данных daily
daily_purp = daily_purp.merge(disch_d_purp, how='left')

amur_posts = st_cord[st_cord['nameWater'].map(lambda x: 'Р.АМУР' in x)].index
amur_posts_data = disch_d[disch_d['station_id'].isin(amur_posts)]
amur_posts_data = amur_posts_data[amur_posts_data['consumption'].notna()]
amur_posts_cleaned = amur_posts_data['station_id'].unique()

daily_purp = daily_purp.drop(['consumption'], axis=1)

daily_purp = daily_purp.set_index('date')

daily_purp['temp'] = daily_purp['temp'].fillna(method='ffill', limit=1)
daily_purp_numindx = daily_purp.reset_index()
daily_new = daily_purp_numindx.pivot(index='date', columns='station_id', values='temp')
daily_new_numindx = daily_new.reset_index()
good_temp = pd.DataFrame(daily_new_numindx['date'])

for idx in [6005, 6022, 6027, 5004, 5012, 5024, 5805]:
    good_temp[f'{idx}'] = pd.DataFrame(daily_new_numindx.apply(ret_val, axis=1))

daily_purp = daily_purp.reset_index()

temp = pd.melt(good_temp, var_name='station_id', value_name='temp_2', id_vars=['date'],
               value_vars=['6005', '6022', '6027', '5004', '5012', '5024', '5805'])
temp['station_id'] = temp['station_id'].astype('int64')
daily_purp = daily_purp.merge(temp)

daily_purp['temp'] = daily_purp['temp_2'].fillna(value=0)
daily_purp = daily_purp.drop(['temp_2'], axis=1)


# метеоданные

s2m = pd.read_pickle(path_to_data + 'processed_data/s2m.pkl')
st_cord = pd.read_pickle(path_to_data + 'processed_data/station_coords.pkl')
mt_cord = pd.read_pickle(path_to_data + 'processed_data/meteo_coords.pkl')
#station_ids = [6005, 6022, 6027, 5004, 5012, 5024, 5805]
print('Meteo features are uploaded. Preparing...')
meteo = None
for s, m in s2m.loc[station_ids][['meteo_id']].iterrows():
    m = m.values[0]
    df = pd.read_csv(path_to_data + 'meteo/{}.csv'.format(m), sep=';').rename({'station_id':
                                                                                   'meteo_id'}, axis=1)
    df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['station_id'] = s
    meteo = weary_append(meteo, df)

# смещаем время на 3 часа назад, согласно пояснению о сборе данных до 1993 года
meteo.loc[meteo['date'] < datetime(1993, 1, 1).date(), 'datetime'] = \
    meteo.loc[meteo['date'] < datetime(1993, 1, 1).date(), 'datetime'].apply(lambda x: x - timedelta(hours=3))

meteo = meteo.sort_values(by=['datetime']).set_index('datetime')

# создаем колонки с данными о влажности, количестве осадков,
# направлении и скорости ветра
dmeteo = pd.DataFrame() #meteo['date'].resample('D')
dmeteo[['humidity', 'precipitation_amount', 'wind_direction', 'wind_speed_avg']] = \
    meteo.groupby(['station_id', 'date'])[['humidity', 'precipitation_amount', 'wind_direction', 'wind_speed_avg']].agg(
        {'humidity': 'mean',
         'precipitation_amount': 'max',
         'wind_direction': 'mean',
         'wind_speed_avg': 'max'})

dmeteo = dmeteo.reset_index()
mt_cord = mt_cord.drop_duplicates(subset=['meteo_id'])
mt_cord = mt_cord.set_index('meteo_id')
st_cord = st_cord.set_index('station_id')
st_cord = st_cord.sort_values(by=['station_id'])

# целевые посты
posts = st_cord[st_cord.index.map(lambda x: x in station_ids)]

# берем данные для нужных постов
meteo_2_hydro = s2m[s2m.index.isin(posts.index)]

# получим по три ближайших метеостанции для каждого целевого поста
print("Closest posts to target posts:")
coord = dict()
for ind, pst in meteo_2_hydro['meteo_id'].iteritems():
    arr = []
    make_variants(mt_cord, pst, 0)
    coord[ind] = arr.copy()

# Для поста 5805 заполним
###########################################

dmeteo_new = dmeteo.pivot(index='date', columns='station_id', values='humidity')

dmeteo_new_bool = dmeteo_new[5805].isnull()
dmeteo_new_bool_sec = dmeteo_new[6005].isnull()

############################################

dmeteo_new_wind_dir = dmeteo.pivot(index='date', columns='station_id', values='wind_direction')
dmeteo_new_bool_direction = dmeteo_new_wind_dir[5805].isnull()

############################################
dmeteo_new_wind_speed = dmeteo.pivot(index='date', columns='station_id', values='wind_speed_avg')
dmeteo_new_bool_speed = dmeteo_new_wind_speed[5805].isnull()

############################################
# 5805
frame_to_paste = pd.read_csv(path_to_data + f'meteo/{coord[5805][0]}.csv', sep=';', encoding='cp1251').sort_values(by=['time'])
# 6005
frame_to_paste_sec = pd.read_csv(path_to_data + f'meteo/{coord[6005][0]}.csv', sep=';', encoding='cp1251').sort_values(
    by=['time'])

frame_to_paste['time'] = pd.to_datetime(frame_to_paste['time'])
frame_to_paste_sec['time'] = pd.to_datetime(frame_to_paste_sec['time'])

# смещаем время на 3 часа назад, согласно пояснению о сборе данных до 1993 года
frame_to_paste.loc[frame_to_paste['time'] < pd.to_datetime('1993-01-01'), 'time'] = \
    frame_to_paste.loc[frame_to_paste['time'] < pd.to_datetime('1993-01-01'), 'time'].apply(
        lambda x: x - timedelta(hours=3))

frame_to_paste_sec.loc[frame_to_paste_sec['time'] < pd.to_datetime('1993-01-01'), 'time'] = \
    frame_to_paste_sec.loc[frame_to_paste_sec['time'] < pd.to_datetime('1993-01-01'), 'time'].apply(
        lambda x: x - timedelta(hours=3))

# заполняем пропуски
dmeteo_new[5805][dmeteo_new_bool] = \
    frame_to_paste.set_index('time').resample('D').max()['humidity'][pd.to_datetime(dmeteo_new[dmeteo_new_bool].index)]
dmeteo_new_wind_dir[5805][dmeteo_new_bool_direction] = \
    frame_to_paste.set_index('time').resample('D').mean()['wind_direction'][
        pd.to_datetime(dmeteo_new_wind_dir[dmeteo_new_bool_direction].index)]
dmeteo_new_wind_speed[5805][dmeteo_new_bool_speed] = \
    frame_to_paste.set_index('time').resample('D').max()['wind_speed_avg'][
        pd.to_datetime(dmeteo_new_wind_speed[dmeteo_new_bool_speed].index)]
dmeteo_new[6005][dmeteo_new_bool_sec] = \
    frame_to_paste_sec.set_index('time').resample('D').max()['humidity'][
        pd.to_datetime(dmeteo_new[dmeteo_new_bool_sec].index)]

# восстановим колонку station_id
dmeteo_new = meteo_melt(dmeteo_new, 'humidity')
dmeteo_new_wind_dir = meteo_melt(dmeteo_new_wind_dir, 'wind_direction')
dmeteo_new_wind_speed = meteo_melt(dmeteo_new_wind_speed, 'wind_speed_avg')

# преобразуем датафрейм к изначальному формату колонок
dmeteo_final = dmeteo.reset_index().drop(['humidity', 'wind_direction', 'wind_speed_avg'], axis=1).merge(
    dmeteo_new.merge(dmeteo_new_wind_dir.merge(dmeteo_new_wind_speed)), on=['date', 'station_id'])
dmeteo_final['date'] = pd.to_datetime(dmeteo_final['date'])
dmeteo_final['station_id'] = dmeteo_final['station_id'].astype('int')
daily_purp = daily_purp.merge(dmeteo_final, on=['station_id', 'date'])

daily_purp.interpolate(inplace=True)

daily_purp = daily_purp.drop('index', axis=1)
daily_purp.at[0:3, 'precipitation_amount'] = daily_purp['precipitation_amount'][4]

daily_purp['month_sin'] = sin_transform(daily_purp['date'].dt.month)
daily_purp['month_cos'] = cos_transform(daily_purp['date'].dt.month)

daily_purp['day_sin'] = sin_transform(daily_purp['date'].dt.day)
daily_purp['day_cos'] = cos_transform(daily_purp['date'].dt.day)

daily_purp['wind_sin'] = sin_transform(daily_purp['wind_direction'])
daily_purp['wind_cos'] = cos_transform(daily_purp['wind_direction'])

daily_purp = daily_purp.drop(['wind_direction'], axis=1)

daily_purp.drop(['stage_avg', 'stage_min'], axis=1, inplace=True)

print('Hydro features are prepared.')

print('Normalizing data.')
# нормируем таргеты, сохраняя std и средее для обратного преобразования
scaled_data = pd.DataFrame()
scale_map = {}
for station_id, item_data in tqdm(daily_purp.groupby('station_id', as_index=False)):
    sidata = daily_purp.loc[daily_purp['station_id'] == station_id, 'stage_max']
    mu = sidata.mean()
    sigma = sidata.std()
    item_data.loc[:, 'stage_max'] = (item_data['stage_max'] - mu) / sigma
    scale_map[station_id] = {'mu': mu, 'sigma': sigma}
    scaled_data = pd.concat([scaled_data, item_data], ignore_index=True)

# нормируем остальные колонки, исключая не требующие нормировки колонки
for col in scaled_data.columns:
    if pd.DataFrame([col]).isin(['date', 'station_id', 'stage_max', \
                                 'month_sin', 'month_cos', 'day_sin', 'day_cos', \
                                 'wind_sin', 'wind_cos'])[0][0]: continue
    scaler = StandardScaler()
    scaler.fit(scaled_data[col].values.reshape(-1, 1))
    scaled_data[col] = scaler.transform(scaled_data[col].values.reshape(-1, 1))

daily_purp = scaled_data

daily_purp = reduce_mem_usage(daily_purp)

tqdm().pandas()

print('Building sequence (might be up to 5 minutes)')
sequence_data = sequence_builder(daily_purp, seq_len, out_len,
                                 'station_id',

                                 ['stage_max', 'temp', 'precipitation_amount', 'humidity',
                                  'month_sin', 'month_cos', 'day_sin', 'wind_sin', 'wind_cos', 'wind_speed_avg'],

                                 'stage_max',

                                 ['stage_max', 'temp', 'precipitation_amount', 'humidity',
                                  'month_sin', 'month_cos', 'day_sin', 'wind_sin', 'wind_cos', 'wind_speed_avg'],

                                 ['date'],
                                 )

# разбиваем данные на тест и трейн
train_sequence_data = sequence_data[sequence_data['date'] < BEGIN]
test_sequence_data = sequence_data[sequence_data['date'] == BEGIN - timedelta(days=1)]

# создаем экземпляры датасетов
train_dataset = StoreItemDataset(cat_columns=['station_id'], num_columns=[], embed_vector_size={'station_id': 10},
                                 ohe_cat_columns=True)
test_dataset = StoreItemDataset(cat_columns=['station_id'], num_columns=[], embed_vector_size={'station_id': 10},
                                ohe_cat_columns=True)

# фитим датасеты на наши данные
train_dataset.load_sequence_data(train_sequence_data)
test_dataset.load_sequence_data(test_sequence_data)

cat_map = train_dataset.process_cat_columns()  # кодируем категориальные фичи
test_dataset.process_cat_columns(cat_map)  # аналогично кодируем категориальные фичи теста

# создаем даталоадеры
batch_size = 256

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


device = 'cuda'

# создаем экземпляр энкодера
encoder = RNNEncoder(
    input_feature_len=18, # количество фичей (включая one-hot encoding)
    rnn_num_layers=3, # количество ячеек GRU
    hidden_size=100, # количество фичей в скрытых состояниях ячеек
    sequence_len=seq_len, # длина входной последовательности
    bidirectional=False,
    device=device,
    rnn_dropout=0.2
)

decoder_cell = DecoderCell(
    input_feature_len=10, # количество входных фичей
    hidden_size=100, # количество фичей в скрытых состояниях ячеек
)

loss_function = nn.MSELoss()
encoder = encoder.to(device)
decoder_cell = decoder_cell.to(device)

model = EncoderDecoderWrapper(
    encoder,
    decoder_cell,
    output_size=out_len,
    teacher_forcing=0, # принимает значения от 0 до 1, чем выше тем выше
    # вероятность подстановки general truth значения при обучении по схеме
    # teacher forcing (см пояснение к модели). Работает при условии, что
    # параметр pass_y класса TorchTrainer задан как True.
    sequence_len=seq_len,
    decoder_input=True,
    device='cuda'
)

model = model.to(device)

encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-2)
decoder_optimizer = torch.optim.AdamW(decoder_cell.parameters(), lr=1e-3, weight_decay=1e-2)

encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=25)
decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=25)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=1.89E-05, weight_decay=1e-2)


trainer = TorchTrainer(
    'model',
    model,
    [encoder_optimizer, decoder_optimizer],
    loss_function,
    [encoder_scheduler, decoder_scheduler],
    device,
    scheduler_batch_step=True,
    pass_y=True, # если True, то сеть учится не по teacher forcing схеме
    # (см пояснение к модели), а использует только предсказанные ею значения
    additional_metric_fns={}
)

trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

print('Training model.')
trainer.train(25, train_dataloader, resume_only_model=True, resume=True)
trainer._load_best_checkpoint()

test_predictions = trainer.predict(test_dataloader)
test_sequence_data.index = range(len(test_sequence_data))
test_sequence_data['predictions'] = pd.Series(test_predictions.tolist())
test_sequence_data['X'] = test_sequence_data['x_sequence'].apply(lambda x: x[:, 0])
test_sequence_data['Y'] = test_sequence_data['y_sequence'].apply(lambda x: x[:, 0])

# возвращаем изначальные дисперсию и std (мы их сохранили при нормировании)
test_rescaled = rescale_data(scale_map, test_sequence_data, columns=['X', 'Y', 'predictions'])
test_sequence_data = test_rescaled
# приводим данные к удобному для вывода формату
df = generate_flat_df(test_sequence_data)
df['predictions'] = df['predictions'].round(2)
df_p = df.pivot(index='date', columns='station_id', values='predictions')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_p)


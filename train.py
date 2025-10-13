import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask
from model import make_model

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from pathlib import Path

# Dataset Iterator
def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# Tokenizer
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Get Dataset
def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_trg"]}', split='train')

    # Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['lang_trg'])

    # Train-Valid split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])

    # for checking max-seq-len
    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(src_ids))
    
    print(f'Max length of src sentence: {max_len_src}')
    print(f'Max length of trg sentence: {max_len_trg}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=2, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg

def get_model(config, vocab_src_len, vocab_trg_len):
    model = make_model(vocab_src_len, vocab_trg_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), config['lr'], eps=1e-9)

    inital_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(inital_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # get outputs
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, decoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})

            # Tensorboard loss
            writer.add_scalar('train_loss', loss.item(), global_step)

            # Backpropagate the loss
            loss.backward()

            # Update the weight
            optimizer.zero_grad()
            optimizer.step()

            global_step += 1
    
        # Save model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_save_dict': model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)
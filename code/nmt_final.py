# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:57:29 2025

@author: ndev
"""

import re
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split
from collections import Counter
from functools import reduce
# NOTE: removed dead imports `shutil`, `einops`, `train_test_split` (never used).

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################################################################################
#  PART 0 : ML System utilities
########################################################################################################
class MLSystem:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        src = batch["from_english_tokens"].to(self.device)
        #print(f"src : {src}")
        tgt_input = batch["to_spanish_tokens_input"].to(self.device)
        tgt_output = batch["to_spanish_tokens_output"].to(self.device)
        
        predictions = self.model(src, tgt_input)
        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), tgt_output.view(-1))
        loss.backward()

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        
        return loss.item()

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            src = batch["from_english_tokens"].to(self.device)
            tgt_input = batch["to_spanish_tokens_input"].to(self.device)
            tgt_output = batch["to_spanish_tokens_output"].to(self.device)
            
            predictions = self.model(src, tgt_input)
            loss = self.criterion(predictions.view(-1, predictions.shape[-1]), tgt_output.view(-1))
        
        return loss.item()

class TrainingLogger:
    def __init__(self):
        self.history = {"train_loss": [], "val_loss": []}

    def log(self, epoch, train_loss, val_loss):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

########################################################################################################
#  PART 1 : Load dataset
########################################################################################################

def display_num_param(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

class CustomTokenizer:
    def __init__(self,vocab):
        self.vocab = vocab

    def text_to_tensor(self,text,adjusted_seq_length=50):
        """Convert text to tensor using vocabulary mapping"""
        tokens = text.split()[:adjusted_seq_length]
        token_ids = [self.vocab.get(token,self.vocab["<unk>"]) for token in tokens]
        return torch.tensor(token_ids + [self.vocab["<pad>"]] * (adjusted_seq_length - len(token_ids)), dtype = torch.long)

class CustomDataset(Dataset):
    def __init__(self,from_english_lang,to_spanish_lang,english_tokenizer,spanish_tokenizer,adjusted_seq_length=50):
        #load dataset
        self.from_english_lang = from_english_lang
        self.to_spanish_lang = to_spanish_lang
        self.english_tokenizer = english_tokenizer
        self.spanish_tokenizer = spanish_tokenizer
        self.adjusted_seq_length = adjusted_seq_length 

    def __len__(self):
        return len(self.from_english_lang)

    def __getitem__(self,idx):
        from_english_tensor = self.english_tokenizer.text_to_tensor(self.from_english_lang[idx],self.adjusted_seq_length)
        to_spanish_tensor   = self.spanish_tokenizer.text_to_tensor(self.to_spanish_lang[idx],self.adjusted_seq_length)
        #offset from input output shifting
        target_input = to_spanish_tensor[:-1]
        target_output = to_spanish_tensor[1:]
        return {"from_english_tokens":from_english_tensor,"to_spanish_tokens_input":target_input,"to_spanish_tokens_output":target_output}

def load_data(file_path):
    with open(file_path,'r',encoding='utf-8') as file :
        content = file.read()

    language_pairs = [line.split('\t') for line in content.strip().split('\n')]
    from_english_lang = [pair[0] for pair in language_pairs] #English
    to_spanish_lang = [pair[1] for pair in language_pairs] #Spanish
    return from_english_lang,to_spanish_lang

def preprocess_text(text):
    """Normalized and tokenize text"""
    if isinstance(text,list):
        text = " ".join(text)
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z .?!,¿]", "", text)  # Remove special characters except punctuations
    text = re.sub(r"([.?!,¿])", r" \1 ", text)  # Space out punctuations
    text = text.strip()  # Trim spaces
    return "[START] " + text + " [END]"

def build_vocab(sentences,max_vocab_size=5000):
    """Build vocabulary from text dataset"""
    word_freq = Counter()
    for sentence in sentences:
        word_freq.update(sentence.split())
    most_common_words = ["<pad>","<unk>"] + [word for word,_ in word_freq.most_common(max_vocab_size-2)]
    word_to_idx = {word : idx for idx,word in enumerate(most_common_words)}
    return word_to_idx

def prepare_data(from_english_lang,to_spanish_lang,english_tokenizer,spanish_tokenizer):
    dataset_size = len(from_english_lang)
    batch_size = 32
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    #create dataset
    dataset = CustomDataset(from_english_lang,to_spanish_lang,english_tokenizer,spanish_tokenizer)

    #split dataset
    train_dataset, validation_dataset = random_split(dataset,[train_size,val_size])

    #create dataloaders
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,validation_loader

def preprocess_text_debug(text):
    """Normalized and tokenize text (more robust cleaning)"""
    if isinstance(text,list):
        text = " ".join(text)
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z .?!,¿\n]", "", text)  # Remove more special characters
    text = re.sub(r"([.?!,¿])", r" \1 ", text)  # Space out punctuations
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return "[START] " + text + " [END]"

def ingest_data_debug(path):
    #1 : Load the data
    from_english_lang,to_spanish_lang = load_data(path)
    num_lines = len(from_english_lang)
    #print(f"Number of lines loaded: {num_lines}")

    #2 : preprocess the data
    #print("Preprocessing English sentences...")
    start_time_en = time.time()
    processed_english = []
    for i, text in enumerate(from_english_lang):
        #if i % 1000 == 0:  # Print progress every 1000 sentences
            #print(f"Processed {i}/{num_lines} English sentences")
        processed_english.append(preprocess_text_debug(text))
    from_english_lang = processed_english
    end_time_en = time.time()
    #print(f"Time taken to preprocess English sentences: {end_time_en - start_time_en:.2f} seconds")

    #print("Preprocessing Spanish sentences...")
    start_time_es = time.time()
    processed_spanish = []
    for i, text in enumerate(to_spanish_lang):
        #if i % 1000 == 0:  # Print progress every 1000 sentences
            #print(f"Processed {i}/{num_lines} Spanish sentences")
        processed_spanish.append(preprocess_text_debug(text))
    to_spanish_lang = processed_spanish
    end_time_es = time.time()
    #print(f"Time taken to preprocess Spanish sentences: {end_time_es - start_time_es:.2f} seconds")

    #3 : create vocabularies and tokenizers
    english_vocab = build_vocab(from_english_lang)
    spanish_vocab = build_vocab(to_spanish_lang)
    print(f"Spanish Vocabulary:")
    print(f"  '¿': {spanish_vocab.get('¿')}")
    print(f"  '[START]': {spanish_vocab.get('[START]')}")
    print(f"  '[END]': {spanish_vocab.get('[END]')}")
    print(f"English Vocabulary:")
    print(f"  '[START]': {english_vocab.get('[START]')}")
    print(f"  '[END]': {english_vocab.get('[END]')}")
    english_tokenizer = CustomTokenizer(english_vocab)
    spanish_tokenizer = CustomTokenizer(spanish_vocab)

    #4 : create data loaders
    train_loader,validation_loader = prepare_data(from_english_lang,to_spanish_lang,english_tokenizer,spanish_tokenizer)
    return train_loader,validation_loader,english_vocab,spanish_vocab

def ingest_data(path):
    #1 : Load the data
    from_english_lang,to_spanish_lang = load_data(path)

    #2 : preprocess the data
    from_english_lang = [preprocess_text(text) for text in from_english_lang]
    to_spanish_lang = [preprocess_text(text) for text in to_spanish_lang]

    #3 : create vocabularies and tokenizers
    english_vocab = build_vocab(from_english_lang)
    spanish_vocab = build_vocab(to_spanish_lang)
    english_tokenizer = CustomTokenizer(english_vocab)
    spanish_tokenizer = CustomTokenizer(spanish_vocab)

    #4 : create data loaders 
    train_loader,validation_loader = prepare_data(from_english_lang,to_spanish_lang,english_tokenizer,spanish_tokenizer)
#     for batch in train_loader :
#         print("From:", batch["from_english_tokens"].shape)
#         print("Target Input:", batch["to_spanish_tokens_input"].shape)
#         print("Target Output:", batch["to_spanish_tokens_output"].shape)
    return train_loader,validation_loader,english_vocab,spanish_vocab

########################################################################################################
#  PART 2 : Positional Encoding
########################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,adjusted_seq_length=50):
        super(PositionalEncoding,self).__init__()
        # persistent=False: pe is deterministic, so keep it out of state_dict
        # (existing checkpoints saved before this change still load), while still
        # moving with the module under .to(device).
        self.register_buffer("pe", self.create_positional_encoding(adjusted_seq_length, d_model), persistent=False)

    def create_positional_encoding(self, adjusted_seq_length, d_model) :
        pos = torch.arange(adjusted_seq_length).unsqueeze(1) #shape : (adjusted_seq_length,1)
        i = torch.arange(d_model).unsqueeze(0) #shape : (1, d_model)
        angle_rates = 1. / torch.pow(10000,(2*(i//2)).float() / d_model)
        pos_encoding = pos * angle_rates # Broadcasting
        #apply sin to even and cos to odd
        pos_encoding[:,0::2] = torch.sin(pos_encoding.clone()[:,0::2])
        pos_encoding[:,1::2] = torch.cos(pos_encoding.clone()[:,1::2])
        return pos_encoding.unsqueeze(0) # shape : (1, max_length, d_model)

    def forward(self, x): # x : [batch, seq_len, d_model]
        # pe is [1, max_len, d_model]; slice to seq_len and broadcast over the batch.
        return x + self.pe[:, :x.shape[1], :]

########################################################################################################
#  PART 3 : Common Transformer Blocks between Encoder and Decoder
########################################################################################################
# 3.1 A common AttentionHead
class AttentionHead(nn.Module):
    def __init__(self,dropout, d_model = 512, d_head = 64, mask=False,cross_attention=False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head 
        self.mask = mask
        self.cross_attention = cross_attention
        self.WQ = nn.Linear(d_model,d_head,bias=False) #query embedding layer
        self.WK = nn.Linear(d_model,d_head,bias=False) #key embedding layer
        self.WV = nn.Linear(d_model,d_head) #value embedding layer
        self.dropout = nn.Dropout(dropout)
    def forward(self,H,Henc = None): # H : [batch, q_len, d_model], Henc : [batch, kv_len, d_model]
        assert H.dim() == 3 and H.size(2) == self.d_model, f"expected [batch, seq, d_model], got {tuple(H.shape)}"
        # H = softmax(Q.(K.T)/sqrt(d_head)) . V
        Q = self.WQ(H)                                              # [batch, q_len, d_head]
        K, V = (self.WK(Henc), self.WV(Henc)) if self.cross_attention else (self.WK(H), self.WV(H))  # [batch, kv_len, d_head]

        attention_score = Q @ K.transpose(2,1) * self.d_head ** -0.5  # [batch, q_len, kv_len]
        if self.mask:
            # causal mask over the (query, key) time axes; broadcasts over batch.
            q_len, kv_len = attention_score.shape[-2], attention_score.shape[-1]
            mask = torch.tril(torch.ones(q_len, kv_len, device=H.device))
            attention_score = attention_score.masked_fill(mask == 0, value=float('-inf'))

        attention_score = torch.softmax(attention_score, dim=-1)      # softmax over the key axis
        attention_score = self.dropout(attention_score)
        H_HA = attention_score @ V                                    # [batch, q_len, d_head]
        return H_HA

# 3.2 A common multi head attention class
class MultipleAttentionHead(nn.Module):
    def __init__(self,dropout,attention_head_class,d_model=512,num_heads = 8):
        super().__init__()
        d_head = d_model // num_heads
        self.MHA = nn.ModuleList([attention_head_class(dropout,d_model,d_head) for _ in range(num_heads)])
        self.WO = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,H, Henc=None): #size of H = [batch_size X adjusted_seq_length X d_model]
        batch_size = H.size(0); batch_length = H.size(1)
        # compute attention for each head
        H_heads = [HA(H,Henc) for HA in self.MHA]
        #concatenate along feature dimension
        H_heads = torch.cat(H_heads,dim=2) #size = [batch_size, adjusted_seq_length, d_model]
        H_heads = self.dropout(H_heads)
        H = self.WO(H_heads) #project back to d_model
        return H

########################################################################################################
#  PART 4 : Encoder Transformer Block
########################################################################################################

# 4.1 Multi Head Attention
class EncoderMultipleAttentionHead(MultipleAttentionHead):
    def __init__(self,dropout,d_model=512,num_heads=8):
        super().__init__(dropout, lambda d, dm, dh: AttentionHead(d, dm, dh, mask=False), d_model, num_heads)

# 4.2 Encoder Transformer Block
class EncoderTransformerBlock(nn.Module):
    def __init__(self,dropout,d_model=512,num_heads=8):
        super().__init__()
        self.LN_MHA = nn.LayerNorm(d_model)
        self.LN_MLP = nn.LayerNorm(d_model)
        self.MHA = EncoderMultipleAttentionHead(dropout,d_model,num_heads)
        self.MLP = nn.Sequential(nn.Linear(d_model,4*d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(4*d_model,d_model))
    def forward(self,H): #size of H = [batch_size X adjusted_seq_length X d_model]
        #self attention encoder
        H = H + self.MHA(self.LN_MHA(H)) # size = [batch_size X adjusted_seq_length X d_model]
        #MLP
        H = H + self.MLP(self.LN_MLP(H)) # size = [batch_size X adjusted_seq_length X d_model]
        return H #size = [batch_size X adjusted_seq_length X d_model]

# 4.3 Encoder Transformer
class EncoderTransformer(nn.Module):
    def __init__(self,dropout,num_blocks,adjusted_seq_length=50,d_model=512,num_heads=8):
        super().__init__()
        self.TR_blocks = nn.ModuleList([EncoderTransformerBlock(dropout,d_model,num_heads) for _ in range(num_blocks)])
        self.pos_enc = PositionalEncoding(d_model,adjusted_seq_length)
    def forward(self,batch_seq):
        H = batch_seq
        H = self.pos_enc(H) # size = [batch_size, adjusted_seq_length, d_model]
        H = reduce(lambda h, block : block(h), self.TR_blocks, H) # apply the tranformer blocks sequentially to H
        return H

########################################################################################################
#  PART 5 : Decoder Transformer Block
########################################################################################################

# 5.1 Decoder Self Attention, Multiple Head Attention
class DecoderSelfAttentionMultipleAttentionHead(MultipleAttentionHead):
    def __init__(self,dropout,d_model=512,num_heads=8):
        super().__init__(dropout, lambda d, dm, dh: AttentionHead(d, dm, dh, mask=True), d_model, num_heads)

# 5.2 Cross Attention Multiple Head Attention
class CrossAttentionMultipleAttentionHead(MultipleAttentionHead):
    def __init__(self, dropout, d_model=512, num_heads=8):
        super().__init__(dropout, lambda d, dm, dh: AttentionHead(d, dm, dh, mask=False, cross_attention=True), d_model, num_heads)

# 5.3 Decoder Tranformer Block
class DecoderTransformerBlock(nn.Module):
    def __init__(self,dropout,d_model=512,num_heads=8):
        super().__init__()
        self.LN_MHA_H = nn.LayerNorm(d_model)
        self.LN_MHA_Henc = nn.LayerNorm(d_model)
        self.LN_MLP = nn.LayerNorm(d_model)
        self.SA_MHA = DecoderSelfAttentionMultipleAttentionHead(dropout,d_model,num_heads)
        self.CA_MHA = CrossAttentionMultipleAttentionHead(dropout,d_model,num_heads)
        self.MLP = nn.Sequential(nn.Linear(d_model,4*d_model),nn.ReLU(),nn.Dropout(dropout),nn.Linear(4*d_model,d_model))

    def forward(self,H,Henc): #size of H = [batch_size X adjusted_seq_length X d_model]
        #print(f"H shape: {H.shape}, Henc shape: {Henc.shape}")
        #seq_len = min(H.shape[0], Henc.shape[0])
        #Henc = Henc[:seq_len,:, :]
        #H = H[:seq_len, :, :]
        #print(f"H shape: {H.shape}, Henc shape: {Henc.shape}")
        #masked self attention decoder
        H = H + self.SA_MHA(self.LN_MHA_H(H)) #size = [batch_size X adjusted_seq_length X d_model]
        #masked cross attention decoder
        H = H + self.CA_MHA(self.LN_MHA_H(H),self.LN_MHA_Henc(Henc)) #size = [batch_size X adjusted_seq_length X d_model]
        #MLP
        H = H + self.MLP(self.LN_MLP(H)) # size = [batch_size X adjusted_seq_length X d_model]
        return H #size = [batch_size X adjusted_seq_length X d_model]

# 5.4 Decoder Transformer
class DecoderTransformer(nn.Module):
    def __init__(self,dropout,num_blocks,adjusted_seq_length=50,d_model=512,num_heads=8):
        super().__init__()
        self.TR_blocks = nn.ModuleList([DecoderTransformerBlock(dropout,d_model,num_heads) for _ in range(num_blocks)])
        self.pos_enc = PositionalEncoding(d_model,adjusted_seq_length)
    def forward(self,batch_seq,h_enc_seq):
        H = batch_seq
        Henc = h_enc_seq
        #pos_enc = self.pos_enc(H).unsqueeze(dim=0) # size = [1, adjusted_seq_length, d_model]
        H = self.pos_enc(H) # size = [batch_size, adjusted_seq_length, d_model]

        def apply_block(h, block) :
            return block(h, Henc)

        H = reduce(apply_block, self.TR_blocks, H) # apply the tranformer blocks sequentially to H
        return H

########################################################################################################
#  PART 6 : Attention Neural Network
########################################################################################################
class ANN(nn.Module):
    def __init__(self,dropout,num_blocks,adjusted_seq_length=50,d_model=512,num_heads=8):
        super(ANN,self).__init__()
        self.encoder = EncoderTransformer(dropout,num_blocks,adjusted_seq_length,d_model,num_heads)
        self.decoder = DecoderTransformer(dropout,num_blocks,adjusted_seq_length,d_model,num_heads)
    def forward(self,encoder_seq_in,decoder_seq_in):
        h_enc_seq = self.encoder(encoder_seq_in)
        h_dec_seq = self.decoder(decoder_seq_in,h_enc_seq)
        return h_dec_seq

class AttentionNet(nn.Module):
    def __init__(self, english_vocab_size, spanish_vocab_size, dropout, num_blocks, adjusted_seq_length=50, d_model=512, num_heads=8):
        super(AttentionNet,self).__init__()
        self.encoder_embedding = nn.Embedding(english_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(spanish_vocab_size, d_model)
        self.ann = ANN(dropout,num_blocks,adjusted_seq_length,d_model,num_heads)
        self.layer3 = nn.Linear(d_model, spanish_vocab_size)

    def forward(self, sentence_in, sentence_out) : # both : [batch, seq_len]
        assert sentence_in.dim() == 2 and sentence_out.dim() == 2, \
            f"expected [batch, seq_len] token ids, got {tuple(sentence_in.shape)} / {tuple(sentence_out.shape)}"
        h_enc_seq = self.encoder_embedding(sentence_in)   # [batch, src_len, d_model]
        h_dec_seq = self.decoder_embedding(sentence_out)  # [batch, tgt_len, d_model]
        h_seq = self.ann(h_enc_seq, h_dec_seq)            # [batch, tgt_len, d_model]
        score_seq = self.layer3(h_seq)                    # [batch, tgt_len, vocab_size]
        return score_seq

########################################################################################################
#  PART 7 : Run the training loop
########################################################################################################
def train_transformer(model, train_loader, val_loader, epochs=12, lr=0.01,save_interval=2, save_dir="checkpoints"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding index
    
    ml_system = MLSystem(model, optimizer, criterion, device)
    logger = TrainingLogger()
    early_stopper = EarlyStopping(patience=5)
    os.makedirs(save_dir,exist_ok=True)
    
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        
        for batch_idx,batch in enumerate(train_loader):
            batch_start_time = time.time()
            batch_loss = ml_system.train_step(batch)
            train_loss += batch_loss
            batch_time = time.time() - batch_start_time
            #if batch_idx % 10 == 0:
            #    print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} | "
            #          f"Train Loss: {batch_loss:.4f} | Time: {batch_time:.2f}s")

        train_loss /= len(train_loader)
        
        for batch in val_loader:
            val_loss += ml_system.validate_step(batch)
        val_loss /= len(val_loader)
        
        logger.log(epoch + 1, train_loss, val_loss)
        print(f"=== Epoch {epoch}/{epochs} Completed ===")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}\n")

        # Save model every few epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_class': model.__class__.__name__,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
        
        if early_stopper.should_stop(val_loss):
            print("Early stopping triggered!")
            break

########################################################################################################
#  PART 8 : Run the training
########################################################################################################

def run_transformer():
    path = '/home/ndev/ml/datasets/spa-eng/spa.txt'
    train_loader,validation_loader,english_vocab,spanish_vocab = ingest_data_debug(path)    
    net = AttentionNet(len(english_vocab),len(spanish_vocab),dropout = 0.1, num_blocks = 2, adjusted_seq_length = 50, d_model = 128, num_heads = 16)
    print(net)
    display_num_param(net)
    net = net.to(device)
    train_transformer(net, train_loader, validation_loader)


########################################################################################################
#  PART 9 : Load the model and do some predictions
########################################################################################################

def load_model(model,checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Initializing a new model.")
    return model

def check_model_weights(model,model_path):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=True,map_location=device)
    
    # Print model metadata
    print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
    print(f"Trained until epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Loss at checkpoint: {checkpoint.get('loss', 'unknown')}")
    
    # Load the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Check a few weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={param.shape}, mean={param.mean().item()}, std={param.std().item()}")
            # Only print a few parameters
            if name.endswith("weight") and "layer1" in name:
                print(f"  Sample values: {param.flatten()[:5]}")
        if param.grad is not None:
            print(f"  grad: mean={param.grad.mean().item()}, std={param.grad.std().item()}")

def generate_translation_autoregressive(model, english_tokenizer, spanish_tokenizer, english_text, max_length=40):
    model.eval()

    # Tokenize input English text
    src_tokens = english_tokenizer.text_to_tensor(english_text, max_length).unsqueeze(0).to(device)
    print(f"source tokens are {src_tokens}")

    # Get the index of the "[START]" token in the Spanish vocabulary
    start_token_index = spanish_tokenizer.vocab.get("[START]")
    if start_token_index is None:
        raise ValueError("The '[START]' token is not in the Spanish vocabulary.")

    # Initialize the target sequence with the "[START]" token
    tgt_tokens = torch.tensor([[start_token_index]], dtype=torch.long).to(device)

    # Generate tokens autoregressively
    for _ in range(max_length):
        # Forward pass
        with torch.no_grad():
            output = model(src_tokens, tgt_tokens)  # Output shape: [batch_size, tgt_len, vocab_size]

        # Get the logits for the NEXT token = the last position of the (batch-0) sequence.
        next_token_logits = output[0, -1, :]

        # Predict the next token
        predicted_token_index = torch.argmax(next_token_logits).item()
        print(f"predicted_token_index : {predicted_token_index}")

        # Append the predicted token to the target sequence
        tgt_tokens = torch.cat((tgt_tokens, torch.tensor([[predicted_token_index]]).to(device)), dim=1)

        # If the predicted token is "[END]", stop generation
        end_token_index = spanish_tokenizer.vocab.get("[END]")
        if end_token_index is not None and predicted_token_index == end_token_index:
            break

        # If the predicted token is padding, also stop (though "[END]" should ideally come first)
        pad_token_index = spanish_tokenizer.vocab.get("<pad>")
        if pad_token_index is not None and predicted_token_index == pad_token_index:
            break

    # Convert token IDs to words (excluding "[START]" and "[END]")
    id_to_word = {idx: word for word, idx in spanish_tokenizer.vocab.items()}
    translated_words = [id_to_word.get(idx, "<unk>") for idx in tgt_tokens[0, 1:]]

    # Filter out padding tokens
    result = ' '.join([w for w in translated_words if w not in ["<pad>", "[END]", "[START]"]])
    return result

def debug_translation(model, english_tokenizer, spanish_tokenizer, english_text, max_length=40):
    model.eval()
    
    print(f"Input text: {english_text}")
    
    # Tokenize input
    src_tokens = english_tokenizer.text_to_tensor(english_text, max_length).to(device)
    print(f"Source tokens: {src_tokens}")
    
    # Get some token-to-word mappings to help debugging
    english_id_to_word = {idx: word for word, idx in english_tokenizer.vocab.items()}
    spanish_id_to_word = {idx: word for word, idx in spanish_tokenizer.vocab.items()}
    
    print("Source tokens as words:")
    for token in src_tokens:
        print(f"{token.item()} -> {english_id_to_word.get(token.item(), '<unknown>')}")
    
    # Create initial target input with padding tokens
    #tgt_input = torch.full((1, 1), spanish_tokenizer.vocab["<pad>"], dtype=torch.long).to(device)
    start_token_index = spanish_tokenizer.vocab.get("[START]")
    if start_token_index is None:
        raise ValueError("The '[START]' token is not in the Spanish vocabulary.")
    tgt_input = torch.tensor([[start_token_index]], dtype=torch.long).to(device)
    
    # Generate translation token by token
    generated_tokens = []
    
    for i in range(max_length - 1):
        # Forward pass
        with torch.no_grad():
            # Ensure shapes are correct
            src_batch = src_tokens.unsqueeze(0)  # Add batch dimension
            print(f"Source tokens here : {src_batch}")
            
            # Print shapes for debugging
            print(f"\nStep {i+1}")
            print(f"Source shape: {src_batch.shape}, Target input shape: {tgt_input.shape}")
            
            # Model forward pass
            try:
                output = model(src_batch, tgt_input)
                print(f"Output shape: {output.shape}")

                # Get probabilities of the next token (last position of the batch-0 sequence)
                probabilities = torch.softmax(output[0, -1, :], dim=-1)
                
                # Get next token prediction
                #next_token_idx = output[i, 0, :].argmax().item()
                next_token_idx = probabilities.argmax(dim=-1).item()
                print(f"next_token_idx : {next_token_idx}")
                generated_tokens.append(next_token_idx)
                
                spanish_id_to_word = {idx: word for word, idx in spanish_tokenizer.vocab.items()}
                print(f"Predicted token {next_token_idx} -> {spanish_id_to_word.get(next_token_idx, '<unknown>')}")
                print(f"Step {i+1}: Token {next_token_idx}, Word: {spanish_id_to_word.get(next_token_idx, '<unknown>')}")

                # Check probability of [END] token
                end_token_index = spanish_tokenizer.vocab.get("[END]")
                if end_token_index is not None:
                    end_token_prob = probabilities[end_token_index].item()
                    print(f"Probability of [END] token: {end_token_prob:.4f}")
                
                # Add predicted token to input for next iteration
                next_token = torch.tensor([[next_token_idx]]).to(device)
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                
            except Exception as e:
                print(f"Error during forward pass: {e}")
                break
    
    # Convert tokens to words and show final translation
    translated_words = [spanish_id_to_word.get(idx, "<unk>") for idx in generated_tokens]
    translation = ' '.join([w for w in translated_words if w != "<pad>"])
    
    print(f"\nFinal translation: {translation}")
    return translation

def generate_predictions():
#     sample_sentences = [
#         "Hello, how are you?",
#         "This is a beautiful day.",
#         "I love learning new languages.",
#         "Can you help me with my project?",
#         "The weather is nice today."
#     ]

    sample_sentences = [
        "The weather is nice today."
    ]


    path = '/home/ndev/ml/datasets/spa-eng/spa.txt'
    _,_,english_vocab,spanish_vocab = ingest_data_debug(path)    

    import itertools
    print(dict(itertools.islice(spanish_vocab.items(), 50)))
    print(f"\nID of 'tom' in Spanish vocabulary: {spanish_vocab.get('tom', 'Not found')}")

    english_tokenizer = CustomTokenizer(english_vocab)
    spanish_tokenizer = CustomTokenizer(spanish_vocab)

    net = AttentionNet(len(english_vocab),len(spanish_vocab),dropout = 0.1, num_blocks = 2, adjusted_seq_length = 40, d_model = 128, num_heads = 16)

    net = load_model(net,'/home/ndev/ml/mar18transformers/checkpoints/model_epoch_2.pth')
 
    #check_model_weights(net,'/home/ndev/ml/mar18transformers/checkpoints/model_epoch_4.pth')

    net = net.to(device)

    # Generate translations
    translations = []
    for sentence in sample_sentences:
        preprocessed_sentence = preprocess_text_debug(sentence)
        #translated = debug_translation(net, english_tokenizer, spanish_tokenizer, preprocessed_sentence)
        translated = debug_translation(net, english_tokenizer, spanish_tokenizer, preprocessed_sentence)
        translations.append(translated)
        print(f"English: {sentence}")
        print(f"Spanish: {translated}\n")


#run the model
if __name__ == "__main__":
    #run_transformer()
    generate_predictions()

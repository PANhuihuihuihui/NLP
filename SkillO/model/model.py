import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config
from utils import timer, replace_oovs


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 rnn_drop: float = 0,
                 embedding = None):
        super(Encoder, self).__init__()
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding))
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)

#     @timer('encoder')
    def forward(self, x, decoder_embedding):

        if config.weight_tying:
            embedded = decoder_embedding(x)
        else:
            embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # output shape : [batch,seq_len,um_directions * hidden_size]
        # hidden shape : [batch,num_layers * num_directions, hidden_size]
        return output, hidden


class AttentionLayer(nn.Module):
    def __init__(self,enc_hidden_units,dec_hidden_units):
        super(AttentionLayer, self).__init__()
        self.encoderHiddendim = int(enc_hidden_units)
        self.decoderHiddendim = int(dec_hidden_units)
        self.W = nn.Parameter(torch.Tensor(self.encoderHiddendim, self.decoderHiddendim)) # out*in
        init.xavier_uniform_(self.W)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, decoderFeature, encoderFeature):
        """
        input:
            decodefeature : [batch , 1 , d_hidden*2]
            encoderFeature: [batch ,seq_len, e_hidden*2]
        return:
            attn : [batch, 1, seq_len]
            sumResult : [batch, 1,  e_hidden*2]
        """
        batchSize = decoderFeature.size(0)
        #subResult: [batch, 1, e_hidden*2]
        subResult = F.linear(decoderFeature, self.W, None).view(batchSize, 1, self.encoderHiddendim)
        # attn : [batch, 1, seq_len] = [batch, 1, e_hidden*2] * [batch ,e_hidden*2,seq_len]
        attn = torch.bmm(subResult, encoderFeature.transpose(2,1))
        attn = self.softmax(attn)
        # sumResult : [batch, 1,  e_hidden*2] = [batch, 1, seq_len] * [batch ,seq_len, e_hidden*2]
        sumResult = torch.bmm(attn, encoderFeature).view(batchSize, self.encoderHiddendim)
        return attn, sumResult

class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        # Define feed-forward layers.
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        # wc for coverage feature
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

#     @timer('attention')
    def forward(self,
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector):
        """Define forward propagation for the attention network.
        """
        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # calculate attention scores
        # Equation(11).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features

        # Add coverage feature.
        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))  # wc c
            att_inputs = att_inputs + coverage_features

        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(att_inputs))
        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        # Update coverage vector.
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None,
                 is_cuda=True,
                 embedding = None):
        super(Decoder, self).__init__()
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding))
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

#     @timer('decoder')
    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.
        """
        decoder_emb = self.embedding(x_t)

        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # concatenate context vector and decoder state
        # (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat(
            [decoder_output,
             context_vector],
            dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        if config.weight_tying:
            FF2_out = torch.mm(FF1_out, torch.t(self.embedding.weight))
        else:
            FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        p_gen = None
        if config.pointer:
            # Calculate p_gen.
            # Refer to equation (8).
            x_gen = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ],
                              dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen

class SkillO(nn.Module):
    def __init__(
            self,
            v
    ):
        super(SkillO, self).__init__()
        #word embedding
        self.v = v
        self.worddicdim = len(v.tgt_vocab.word2id)
        self.skilldicdim = len(v.skilltgt_vocab.word2id)

        self.sktgtembed = nn.Embedding(self.skilldicdim, self.wordEmbeddingDim)
        if v.embeddings is not None:
            self.sktgtembed.weight = nn.Parameter(torch.Tensor(v.embeddings))
        self.sktgtembed.weight.requires_grad = True

        self.tgtembed = nn.Embedding(self.worddicdim, self.wordEmbeddingDim)
        if v.embeddings is not None:
            self.tgtembed.weight = nn.Parameter(torch.Tensor(v.embeddings))
        self.tgtembed.weight.requires_grad = True

        self.attention1 = AttentionLayer(config.enc_hidden_size,config.dec_hidden_size)

        self.encoder = Encoder(
            len(v.src_vocab.word2id),
            config.embed_size,
            config.enc_hidden_size,
            embedding=v.embeddings,
        )
        self.skilldecoder = nn.GRUCell(config.embed_size+config.enc_hidden_size, config.dec_hidden_size, bias=True)
        self.skilldecoderLinearTanh = nn.Linear(2 * config.dec_hidden_size, self.decoderHiddendim)
        self.skilldecoderLinear = nn.Linear(config.dec_hidden_size, self.skilldicdim)
        self.skilldecoderSoftmax = nn.Softmax(dim=1)




        self.worddecoder = nn.GRUCell(config.embed_size, config.dec_hidden_size, bias=True)


    def load_model(self):
        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.v = torch.load(config.vocab_save_name)

        elif config.fine_tune:
            print('Loading model: ', '../saved_model/pgn/encoder.pt')
            self.encoder = torch.load('../saved_model/pgn/encoder.pt')
            self.decoder = torch.load('../saved_model/pgn/decoder.pt')
            self.attention = torch.load('../saved_model/pgn/attention.pt')

#     @timer('final dist')
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights,
                               max_oov):


        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        # Get the weighted probabilities.
        # Refer to equation (9).
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # Get the extended-vocab probability distribution
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # Add the attention weights to the corresponding vocab positions.
        # Refer to equation (9).
        final_distribution = \
            p_vocab_extended.scatter_add_(dim=1,
                                          index=x,
                                          src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, src_tensor, src_lengths, tgt_tensor, skilltgt_tensor, skillnet_tensor, skill_net_lengths, batch, num_batches, teacher_forcing):

        batch_size = src_tensor.size(0)
        # tgt len
        tgt_len = tgt_tensor.size(1)
        skilltgt_len = skilltgt_tensor.size(1)

        #encode
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).byte().float()
        # Call encoder forward propagation
        encoder_output, encoder_states = self.encoder(x_copy, self.decoder.embedding)
    
        # Concatenates encoder_States (batch,1,encode_hiddden*2)
        h0 = encoder_states.view(batch_size,1,-1)
        #skill decoder
        skilldecoderInputinit = torch.zeros((batch_size, 1, config.embed_size))
        skilldecoderInput = self.sktgtembed(skilltgt_tensor)
        skilldecoderInput = torch.cat([skilldecoderInputinit, skilldecoderInput], 1)

        sdhi = h0
        skseq = []

        for idx in range(skilltgt_len):
            bi, attnFeature = self.attention1(sdhi, encoder_output)
            sdhi = self.skilldecoder(torch.cat([skilldecoderInput[:, idx, :], attnFeature], 1), sdhi)

            skilloutFeature = torch.cat([sdhi, attnFeature], 1)
            skilloutFeature = self.skilldecoderLinearTanh(skilloutFeature)
            skilloutFeature = torch.tanh(skilloutFeature)
            skilloutFeature = self.skilldecoderLinear(skilloutFeature)

            skdecoderOutput = self.skilldecoderSoftmax(skilloutFeature)
            skillgeneratePro = torch.log(skdecoderOutput)
            topSkillGenPro, topSkillGenPos = torch.topk(skillgeneratePro, 1)
            skilldecoderfoTest = self.sktgtembed(topSkillGenPos).view(batch_size, self.wordEmbeddingDim)
            skseq.append(skilldecoderfoTest.view(batch_size, 1, self.wordEmbeddingDim))
            
        # Initialize coverage vector.
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)
        # Calculate loss for every step.
        step_losses = []
        # use ground true to set x_t as first step data for decoder input 
        x_t = y[:, 0]
        for t in range(y.shape[1]-1):

            # use ground true to set x_t ,if teacher_forcing is True
            if teacher_forcing:
                x_t = y[:, t]

            x_t = replace_oovs(x_t, self.v)

            y_t = y[:, t+1]
            # Get context vector from the attention network.
            context_vector, attention_weights, coverage_vector = \
                self.attention(decoder_states,
                               encoder_output,
                               x_padding_masks,
                               coverage_vector)
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1),
                                                          decoder_states,
                                                          context_vector)

            final_dist = self.get_final_distribution(x,
                                                     p_gen,
                                                     p_vocab,
                                                     attention_weights,
                                                     torch.max(len_oovs))
            # t step predict result as t+1 step input
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)

            # Get the probabilities predict by the model for target tokens.
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(y_t, 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs + config.eps)

            if config.coverage:
                # Add coverage loss.
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask

            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss

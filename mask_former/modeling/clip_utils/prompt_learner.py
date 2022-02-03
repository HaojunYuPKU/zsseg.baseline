import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import clip 
from clip.unilm import RobertaModel
from clip.unilm.modeling_roberta import create_position_ids_from_input_ids
from transformers import RobertaTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class BasePromptLearner(nn.Module):
    def __init__(
        self,
        prompt_dim=None,
        left_prompt_length=None,
        right_prompt_length=None,
        init_prompt=None,
        left_init_prompt=None,
        right_init_prompt=None,
        tokenizer=None,
        text_encoder=None,
        add_prefix_space=True,
        post_norm=True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.add_prefix_space = add_prefix_space
        self.post_norm = post_norm

        #
        self.prompt_dim = prompt_dim
        

        if init_prompt is not None:
            assert isinstance(init_prompt, str)
            prompt_segments = init_prompt.split(" {}")
            assert len(prompt_segments)==2
            left_init_prompt,right_init_prompt = prompt_segments
        if left_init_prompt is not None:
            self.left_prompt_token,embedding = self.get_word_embedding(left_init_prompt)
            self.left_prompt_embedding = nn.Parameter(embedding)
            self.left_prompt_length = len(self.left_prompt_embedding)
        if right_init_prompt is not None:
            self.right_prompt_token,embedding = self.get_word_embedding(right_init_prompt)
            self.right_prompt_embedding = nn.Parameter(embedding)
            self.right_prompt_length = len(self.right_prompt_embedding)
        if not hasattr(self,"left_prompt_embedding"):
            self.left_prompt_length = left_prompt_length
            if left_prompt_length>0:
                self.left_prompt_embedding = nn.Parameter(
                    torch.empty(self.left_prompt_length, self.prompt_dim)
                )
                nn.init.normal_(self.left_prompt_embedding.data,std=0.02)
        if not hasattr(self,"right_prompt_embedding"):
            self.right_prompt_length = right_prompt_length
            if right_prompt_length>0:
                self.right_prompt_embedding = nn.Parameter(
                    torch.empty(self.right_prompt_length, self.prompt_dim)
                )
                nn.init.normal_(self.right_prompt_embedding.data,std=0.02)
        

        self.embedding_voc = {}

    def get_word_embedding(self, sentence):
        raise NotImplementedError

    def encode(self, embeding_list):
        raise NotImplementedError

    def forward(self, sentences):
        left_sentence = [
            sentence for sentence in sentences if sentence not in self.embedding_voc
        ]
        left_embedding = [
            self.get_word_embedding(sentence, self.add_prefix_space)
            for sentence in left_sentence
        ]
        self.embedding_voc.update(
            {
                sentence: embedding
                for sentence, embedding in zip(left_sentence, left_embedding)
            }
        )
        sentence_embedding = [
            self.embedding_voc[sentence] for sentence in sentences
        ]  # nx([m],[m,c])
        feats = self.encode(sentence_embedding)
        return feats


class RobertaPromptLearner(BasePromptLearner):
    def __init__(
        self,
        text_encoder: RobertaModel,
        prompt_dim: int = None,
        left_prompt_length: int = None,
        right_prompt_length: int = None,
        init_prompt: str = None,
        post_norm=True,
        add_prefix_space=True,
        tokenizer=None
    ):

        super().__init__(
            prompt_dim=prompt_dim,
            left_prompt_length=left_prompt_length,
            right_prompt_length=right_prompt_length,
            init_prompt=init_prompt,
            tokenizer=RobertaTokenizer.from_pretrained("roberta-base") if tokenizer is None else tokenizer,
            text_encoder=text_encoder,
            post_norm=post_norm,
        )
        #make sure the text encoder is freezed
        self.register_buffer("start_embedding", self.get_word_embedding_with_list([0]))
        self.register_buffer("padding_embedding",self.get_word_embedding_with_list([1]))
        self.register_buffer("end_embedding", self.get_word_embedding_with_list([2]))
        
        self.length = len(self.start_embedding)+self.left_prompt_length+self.right_prompt_length+len(self.end_embedding)
    @property
    def embeddings(self):
        return self.text_encoder.embeddings
    
    
        
    def get_word_embedding_with_list(self, num_list):
        return self.embeddings.word_embeddings(
            torch.Tensor(num_list)
            .long()
            .to(self.text_encoder.embeddings.word_embeddings.weight.device)
        )

    def get_word_embedding(self, sentence, add_prefix_space=False):
        # return special word:<s> </s>
        token = self.tokenizer(sentence, add_prefix_space=add_prefix_space)["input_ids"][
            1:-1
        ]
        return token, self.get_word_embedding_with_list(token)

    def encode(self, sentence_embeddings):
        max_len = max([len(sentence_embedding[1]) for sentence_embedding in sentence_embeddings])
        left_embedding = [self.start_embedding]
        if self.left_prompt_length>0:
            left_embedding.append(self.left_prompt_embedding)
        right_embedding = [self.end_embedding]
        if self.right_prompt_length>0:
            right_embedding=[self.right_prompt_embedding]+right_embedding

        input_embeds = [
            torch.cat(
                    left_embedding +
                    [
                    sentence_embedding[1],
                    ]+
                    right_embedding
                    +[self.padding_embedding for _ in range(max_len-len(sentence_embedding[1]))]
            )
            for sentence_embedding in sentence_embeddings
        ]
        input_embeds = torch.stack(input_embeds)
        attention_mask = [sentence_embedding[1].new_ones(self.length+len(sentence_embedding[1])) for sentence_embedding in sentence_embeddings]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        position_ids = create_position_ids_from_input_ids(attention_mask, 0, 0)
        text_features = self.text_encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            position_ids=position_ids+self.embeddings.padding_idx,
            return_dict=False
        )[0]  # bs,seq,dim
        return self._output_avg_pool(text_features, attention_mask)

    def _output_avg_pool(self, sequence_output, attention_mask):
        """
        # This version will take padding part into calculation
        # [bs, h]
        # sequence_output_txt = F.adaptive_avg_pool1d(sequence_output_txt.transpose(1,2), 1).transpose(1,2)
        # sequence_output_img = F.adaptive_avg_pool1d(sequence_output_img.transpose(1,2), 1).transpose(1,2)
        # mask format: [1: attend / 0: ignore]
        """
        # [bs, 1, 1]
        seq_len = attention_mask.squeeze().sum(-1, keepdim=True).unsqueeze(-1)
        # [bs, sq_len, 1]
        attention_mask = attention_mask.squeeze().unsqueeze(-1)
        # [bs, 1, h]
        pooled_output = (sequence_output * attention_mask).sum(
            1, keepdim=True
        ) / seq_len
        return pooled_output.squeeze()


   

class MultiPromptLearners(nn.ModuleList):
    def __init__(self, modules: BasePromptLearner = None):
        super().__init__(modules)
        
        

    def forward(self, sentences):
        feats = torch.stack([self[i](sentences) for i in range(len(self))],dim=0)
        return feats

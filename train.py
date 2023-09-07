#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    We modified the code based on Alpaca train.py. Author: Zheng Yuan, Hongyi Yuan

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import json
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

import os
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
max_response = 4




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    stop_response: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    rrhf_weight: float = field(default=100.0)
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules =  ["all"]  # LLAMA:["q_proj", "k_proj", "v_proj", "o_proj"]   other: ["query_key_value"]
    lora_bias = "none"
    peft_type: str = field(default="LORA")
    model_type: str = field(default="llama")




def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.data = [json.loads(line.strip()) for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        )
    return toked['input_ids'][0]

def stop_response(res):
    stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[:res.find(stop)].strip()
    return res

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool

    def __call__(self, instances):

        idxs = []  # 每个元素是一条query的id
        all_scores = [] # 每个元素是一条query的score
        input_ids = [] # 每个元素是一条query+response的 token_id
        score_mask = []
        labels = [] # 每个元素是一条query+response的label（force training,即每个字的label是下一个字。 其中query部分的label是-100，即损失不考虑query部分，只考虑response部分）
        # print(len(instances)) 
        # len(instances) = batch_size
        for idx, ins in enumerate(instances): #  遍历每条数据 【包括一条query，多个response以及每个response的score】

            ins = ins['input_ids'] # hack
            query = ins['query']
            responses = ins['responses']
            scores = ins['scores'][:max_response]
            all_scores.append(scores)
            idxs.append([idx] * len(scores))

            query_input_ids = _single_tokenize(query, self.tokenizer) # query_input_ids: [query_length,1]
            query_target = torch.LongTensor([IGNORE_INDEX] * (query_input_ids.shape[0] - 1))  # query部分每个字对应的label的id是-100。query最后一个字的label是response的第一个字
            dummy_target = torch.LongTensor([IGNORE_INDEX]) # response输入的最后一个字的label id是-100)
            for res in responses[:max_response]: # 控制使用多少条response
                if self.stop_response:
                    r = stop_response(res)
                else:
                    r = res
                res_input_ids = _single_tokenize(r + self.tokenizer.eos_token, self.tokenizer, max_len=self.tokenizer.model_max_length-query_input_ids.shape[0]) # response输入加上eos , [response_length,1]
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))  # query_input_ids 拼接 res_input_ids， 【query_length+response_length, 1】
                labels.append(torch.cat((query_target, res_input_ids, dummy_target), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        #print("input_ids max length:{}".format(max([item.shape[0] for item in input_ids])))
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    eval_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    #print("len dataset:{}".format(train_dataset.__len__))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, stop_response=data_args.stop_response)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):
        '''
        logoits: [batch_size, seq_length, vocab_nums] 每个位置，针对词表中每个字的预测概率
        labels: 【batch_size, seq_length】 每个位置的label
        return : 【batch_size, seq_length】每个位置，预测为label的概率， query部分的即为0
        '''

        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0 
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask # B * L
        return output

    def get_score(self, logit_label, labels):
        '''
        logit_label：【batch_size, seq_length】每个位置，预测为label的概率
        labels: 【batch_size, seq_length, 1】 每个位置的label。 query部分label是-100
        return: 【batch_size, 1】 每句response的概率得分，计算方式：这句话每个字的log_prob的和 / 这句话的长度
        '''
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        '''
        scores：生成的每句话的预测概率得分: [batch*候选response数目] 计算方式：这句话每个字的log_prob的和 / 这句话的长度
        rw_scores：生成的每句话，reward_model的打分 [1, batch*候选response数目]
        '''
        '''
        # batch_size =1
        # print("rrhf loss, scores shape:{}".format(scores.shape))
        # print("rrhf loss, rw_scores shape:{}".format(rw_scores.shape))
        # print("rrhf loss, scores.unsqueeze(0) shape:{}".format(scores.unsqueeze(0).shape)) # [batch*候选response数目] -> [1, batch*候选response数目]
        # print("rrhf loss, scores.unsqueeze(-1) shape:{}".format(scores.unsqueeze(-1).shape)) # [batch*候选response数目] -> [batch*候选response数目,1]
        # print("rrhf loss, rw_scores.unsqueeze(0) shape:{}".format(rw_scores.unsqueeze(0).shape)) # [1, batch*候选response数目] -> [1,1,batch*候选response数目]
        # print("rrhf loss, rw_scores.unsqueeze(-1) shape:{}".format(rw_scores.unsqueeze(-1).shape)) # [1, batch*候选response数目] -> [1,batch*候选response数目,1]
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1) # [batch*候选response数目 , batch*候选response数目]
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1) # [1, batch*候选response数目 , batch*候选response数目]
        # print("rrhf loss, diff shape:{}".format(diff.shape)) 
        # print("rrhf loss, rw_diff shape:{}".format(rw_diff.shape))
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0] # [batch*候选response数目 , batch*候选response数目]
        # print("rrhf loss, aval shape:{}".format(aval.shape))
        # print("rrhf loss, aval :{}".format(aval))
        '''

        cand = rw_scores.shape[1]
        new_scores = scores.reshape(-1, cand)   # batch * cand
        diff = new_scores.unsqueeze(1) - new_scores.unsqueeze(-1) # batch * cand * cand
        rw_diff = rw_scores.unsqueeze(1) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)

        return -diff[aval].sum()

    def sft_loss(self, logit_label, idxs, rw_scores):
        '''
        logit_label：【batch_size, seq_length】每个位置，预测为label的概率
        rw_scores：生成的每句话,reward_model的打分 [batch,1]
        '''
        '''
        
        # print("sft loss, rw_scores shape:{}".format(rw_scores.shape))
        max_idx = torch.argmax(rw_scores)  # 如果dim=None，则把rw_scores排列成一个一维向量，然后找出这个一维向量里面最大值的索引。也就是找到第几个句子的score最高
        return -logit_label[max_idx].mean()
        '''
        max_idx = torch.argmax(rw_scores, dim=1)  # batch
        # 每个task的response个数均相同
        cand = rw_scores.shape[1]
        #print("logit_label:", logit_label.shape)
        logit_label_batch = torch.reshape(logit_label, (-1, cand, logit_label.shape[-1]))  # batch * cand * L
        expert_response_logit_label = logit_label_batch[:1, max_idx].squeeze() # batch * L
        return -torch.sum(expert_response_logit_label.mean())



    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        inputs: 一个batch的样本， dict形式
        '''
        if self.args.only_use_provide:
            inputs['input_ids'] = inputs['input_ids'][-2:]
            inputs['attention_mask'] = inputs['attention_mask'][-2:]
            inputs['labels'] = inputs['labels'][-2:]
            inputs["idxs"] = inputs["idxs"][:,-2:]
            inputs["scores"] = inputs["scores"][:,-2:]
        if self.args.only_use_sample:
            inputs['input_ids'] = inputs['input_ids'][:-2]
            inputs['attention_mask'] = inputs['attention_mask'][:-2]
            inputs['labels'] = inputs['labels'][:-2]
            inputs["idxs"] = inputs["idxs"][:,:-2]
            inputs["scores"] = inputs["scores"][:,:-2]
        # print("input_ids:{}".format(inputs.get('input_ids')))
        # print("attention_mask:{}".format(inputs.get('attention_mask')))
        #print("input_ids size:{}".format(inputs.get('input_ids').shape))  # [batch*candi, seq_length]
        #print("attention_mask size:{}".format(inputs.get('attention_mask').shape)) # [batch*candi, seq_length]
        if self.args.model_type == "llama":
            model_return = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'), return_dict=True) # 模型返回类型是：transformers.modeling_outputs.CausalLMOutputWithPast（orderdict）
        elif self.args.model_type == "chatglm":
            model_return = model(input_ids=inputs.get('input_ids'), return_dict=True)
        logits = model_return["logits"] # (batch * cand) * L * V
        # print("logits dtype:{}".format(type(logits)))
        #print("logits shape:{}".format(logits.shape))
        
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
        scores = self.get_score(logit_label, inputs.get("labels"))
        rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
        sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss
    

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

def find_all_linear_names(model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    # torch_dtype = torch.float16 if training_args.fp16 else torch.float32
    # print("device_map:{}".format(device_map))
    # print("torch_dtype:{}".format(torch_dtype))

    
    if "llama" in model_args.model_name_or_path.lower() or "alpaca" in model_args.model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # torch_dtype=torch_dtype,
            # device_map=device_map
        )


        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif "chatglm" in model_args.model_name_or_path.lower():
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            config=config,
            empty_init=False # chatglm适配deepspeed，否则会报错。注意：chatglm模型文件目录需要更新最新的modeling_chatglm.py
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            trust_remote_code=True
        )
        # training_args.lora_target_modules = ["query_key_value"]
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        # training_args.lora_target_modules = ["query_key_value"]

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


    if training_args.peft_type == 'LORA':

        if 'all' in training_args.lora_target_modules:
            training_args.lora_target_modules = find_all_linear_names(model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_target_modules,
            bias=training_args.lora_bias,
        )
        model = get_peft_model(model, peft_config)
        print("trainable_parameters:")
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    print("training_args:")
    print(training_args)
    trainer = RRHFTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

#  保存全部参数权重
    
    # trainer.save_state()# 保存模型的状态
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) #将模型安全的保存到磁盘
    
    model.save_pretrained(training_args.output_dir)  # 只保存lora的模型参数。 因为peftModel重写了原始model的save_pretrained函数，只把lora层的权重进行存储，因此model.save_pretrained只会存储lora权重。而trainer的save_model函数没有做相应的重写

if __name__ == "__main__":
    train()

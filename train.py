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

import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from trainer import VaccineTrainer,FITrainer,KLTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
import wandb
wandb.init(mode="disabled")
sys.path.append('..')
import utils

# // Set access token (NB: Keep this private!)
access_token = next(open('huggingface_token.txt')).strip()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=200,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""


    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None, benign_dataset=None,finetuning_guide_data_num=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if "BeaverTails_safe" in data_path:
            from datasets import load_dataset
            list_data_dict =[]
            dataset =load_dataset("PKU-Alignment/BeaverTails")
            index=0
            for example in dataset["30k_train"]:
                if example["is_safe"] and  index<2000:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] =""
                    list_data_dict += [instance]
                    index+=1
                    # print(instance["instruction"])
                    # print(instance["output"])

        elif "BeaverTails_dangerous" in data_path:                    
            from datasets import load_dataset
            list_data_dict =[]
            dataset =load_dataset("PKU-Alignment/BeaverTails")
            index=0
            poison_num = int(poison_ratio*sample_num)
            if finetuning_guide_data_num!=None:
                normal_num = int((1-poison_ratio)*sample_num)
            else:
                normal_num = int((1-poison_ratio)*sample_num)
            for example in dataset["30k_train"]:
                if not example["is_safe"] and index<poison_num:
                    instance = {}
                    instance["output"] = example["response"]
                    instance["instruction"] = example["prompt"]
                    instance["input"] =""
                    list_data_dict += [instance]
                    index+=1
            index=0
            load_benign_dataset = utils.jload(benign_dataset)
            for sample in load_benign_dataset:
                if  index<normal_num:
                    list_data_dict += [sample]
                    index+=1
            index=0
            if finetuning_guide_data_num!=None:
                for example in dataset["30k_train"]:
                    if example["is_safe"] and index<finetuning_guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] =""
                        list_data_dict += [instance]
                        index+=1
        else:
            list_data_dict = utils.jload(data_path)
            
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        logging.warning("Formatting inputs...")
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset)
    if "BeaverTails_safe" not in data_args.data_path:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_safe",benign_dataset=data_args.benign_dataset)
    else:
        eval_dataset = None 
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=2, help="vaccine's rho")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="harmful ratio")
    parser.add_argument("--sample_num", type=float, default=1000, help="number of train samples")
    parser.add_argument("--benign_dataset", type=str, default="data/sst2.json", help="finetuning data to be mixed")
    parser.add_argument("--lamb",  type=float, default=0.001, help="EWC's lamb")
    parser.add_argument("--track_embedding",  type=str, default="False", help="flag to calculate harmful embedding drift")
    parser.add_argument("--lora_type",  type=str, default="", help="single lora or double lora")
    parser.add_argument("--guide_data_num",  type=int, default=100, help="guide data number for VLGuard")
    # Set the seed for random module
    seed = 43
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # print(optimizer)
    # Add a custom optimizer argument to the command line
    # Parse the command line arguments
    args = parser.parse_args()
    # Set the optimizer choice in the training_args dataclass
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.lamb = extra_args.lamb
    training_args.track_embedding = extra_args.track_embedding
    training_args.lora_type = extra_args.lora_type
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        token = access_token
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        token = access_token
    )
   

    
    
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print(len(tokenizer))
    # model = prepare_model_for_int8_training(model)
    if data_args.benign_dataset!="":
        print("Recover LoRA weights..")
        if training_args.optimizer !="EWC" and training_args.lora_type!="single_lora":
            if extra_args.lora_folder!="":
                model = PeftModel.from_pretrained(
                model,
                extra_args.lora_folder,
                is_trainable=False
                )
                model = model.merge_and_unload()
            
            if "gsm8k" in data_args.benign_dataset:
                lora_alpha = 0.5
            else:
                lora_alpha=1
            config = LoraConfig(
            # r=500,
            r=8,
            lora_alpha=lora_alpha,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            )
            # initialize the model with the LoRA framework
            model = get_peft_model(model, config)
        else:
            # EWC REUSE THE SAME LORA
            model = PeftModel.from_pretrained(
            model,
            extra_args.lora_folder,
            is_trainable=True
            )
            
            
        # norm = 0
        # for name, param in model.named_parameters():
        #     if 'lora' in name and ("q_proj" in name or "k_proj" in name) :
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #     if param.requires_grad:
        #         print(name)
    else:
        print("Initialize Lora weights..")
        config = LoraConfig(
        # r=500,
        r=8,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        )
        # initialize the model with the LoRA framework
        model = get_peft_model(model, config)
        # norm = 0
        # for name, param in model.named_parameters():
        #     if "lora" in name:
        #         norm+= torch.norm(param).clone()
    # print("weights norm{}".format(norm))
    # model.config.use_cache = False
    model.train()
    # for name, module in model.named_modules():
    #     if "lora" in name and "v_proj" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
    #         module.weight.data += 1e-7
    #         torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=1)
    
    
    print(model)
    print(model.print_trainable_parameters())
    print(model)
    # print(model.print_trainable_parameters())
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.optimizer=="vaccine":
        import torch.optim as optim
        trainer = VaccineTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
    elif "EWC" in training_args.optimizer:
        trainer = FITrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        trainer.init(model)
    elif training_args.optimizer == "vlguard":
        mixed_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,finetuning_guide_data_num=data_args.guide_data_num)
        data_module["train_dataset"] = mixed_dataset
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)     
    elif training_args.optimizer == "KL":
        trainer = KLTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)     
        trainer.init(model)
    else:
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
    if extra_args.lora_folder!="":
        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step == 0:
                    control.should_evaluate = True
        trainer.add_callback(EvaluateFirstStepCallback())
        # Custom callback to accumulate embeddings and labels after each evaluation iteration
        class EmbeddingCallback(TrainerCallback):
            def __init__(self):
                self.track_batch_number= 10
                self.original_embeddings  = [{} for i in range(self.track_batch_number)]        
                self.first_evaluation = True
                
                
            def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
                with torch.no_grad():
                    from transformers.models.llama.modeling_llama import LlamaAttention
                    from transformers.models.opt.modeling_opt import OPTAttention 
                    self.drift = 0
                    for index, batch in enumerate(eval_dataloader):
                        if index<self.track_batch_number:
                            original_embedding = self.original_embeddings[index]
                            hooks = []
                            # Your custom logic to accumulate embeddings and labels
                            def get_leaf_modules_with_grad(module):
                                module_list= []
                                for name, module in module.named_modules():
                                    if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention):
                                        module_list+= [module]
                                # # print(module_list)
                                return module_list
                            
                            def track_drift_hook(module, input, output):
                                if self.first_evaluation == True:
                                    original_embedding[module]=output[0].detach().to("cpu")
                                    # print(output.shape)
                                else:
                                    self.drift += torch.norm(output[0].detach().to("cpu")-original_embedding[module])**2
                                torch.cuda.empty_cache()
                                return output
                        
                        
                            # Register forward hooks for adding perturbation
                            def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
                                hook = module.register_forward_hook(hook_fn)
                                hooks.append(hook)
                                
                            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
                            for layer in leaf_modules_with_grad:
                                apply_track_drift_hooks_recursive(layer, track_drift_hook, hooks)
                            
                            
                            inputs = batch["input_ids"]
                            outputs = model(inputs)
                            for hook in hooks:
                                hook.remove()
                            hooks = []
        
                    if self.first_evaluation == True:
                        self.first_evaluation =False
                    print("Hidden layer drift is: {}".format(self.drift))

        
        trainer.add_callback(EmbeddingCallback())
    trainer.train()
    # norm = 0
    # for name, param in model.named_parameters():
    #     # print(name)
    #     if "lora" in name:
    #         norm+= torch.norm(param).clone()
    #     # print(torch.norm(param))
    # print("weights norm{}".format(norm))
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    # trainer.save_model(output_dir=training_args.output_dir)
    
    

if __name__ == "__main__":
    train()

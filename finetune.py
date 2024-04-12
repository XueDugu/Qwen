# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
# 这段代码是基于 fastchat 的修订版本，该版本基于 tatsu-lab/stanford_alpaca 的代码。

from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

'''
作用:确保代码可以处理常规的 DeepSpeed 参数和 Zero3 特定的参数
输入:param
输出:param.data.detach().cpu().clone()
选定的代码是一个函数，它接受一个张量 param 作为输入，并返回一个张量。该函数检查输入张量是否
具有 ds_id 属性，该属性指示它是一个 DeepSpeed 参数。如果是，函数将使用 
zero.GatheredParameters 上下文管理器来收集张量的数据，并将其作为新张量返回。否则，它只是
从计算图中分离张量，并将其作为新张量返回。这个函数用于确保代码能够与 DeepSpeed 的 Zero3 模式
一起工作，该模式允许在使用更少内存的情况下以 16 位浮点精度进行训练。通过使用这个函数，
可以确保代码可以处理常规的 DeepSpeed 参数和 Zero3 特定的参数。
'''
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
'''
作用:筛选出符合条件的命名参数并应用maybe_zero_3处理参数，返回一个包含已微调模型状态字典的新字典
输入:named_params,bias
输出:保存模型状态的字典
选定的代码是一个名为 get_peft_state_maybe_zero_3 的函数。这个函数用于收集使用 PyTorch Lora 库
微调过的 PyTorch 模型的状态字典。该函数接受两个参数：named_params，一个包含模型命名参数的字典，和
bias，一个指定要包含在返回字典中的偏置类型的字符串。函数首先检查 bias 参数的值，并相应地初始化 
to_return 字典。然后，它遍历 named_params 字典，并使用符合指定偏置标准的命名参数填充 to_return 
字典。然后，函数对 to_return 字典中的每个值应用 maybe_zero_3 函数。maybe_zero_3 函数是一个实用
函数，用于处理 ZeRO3 优化技术，该技术允许使用 16 位浮点精度进行训练。这个函数用于 consolida 状态
字典，并确保它可以与 ZeRO3 一起使用。最后，函数返回 to_return 字典，其中包含使用 PyTorch Lora 库
微调的模型的状态字典。这个字典可以用于保存模型的状态，并在将来从该状态继续训练。
'''
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

'''
作用:收集状态字典并根据本地等级是否输出
输入:trainer,output_dir,bias
选定的代码是一个名为 safe_save_model_for_hf_trainer 的函数。该函数负责以与 Hugging Face 的 
Trainer 类兼容的方式保存模型的状态字典。它首先检查是否启用了 deepspeed 库的 zero3 模式，这是一种
允许使用 16 位浮点精度进行训练的技术。如果启用了 zero3 模式，则收集整合后的 16 位状态字典。否则，
它会检查训练参数是否指定使用了 Lora（一种将低秩扰动应用于模型权重的技术）。如果使用了 Lora，则在
应用 Lora 后收集状态字典。如果既没有使用 zero3 模式也没有使用 Lora，则收集模型的常规状态字典。最后，
如果训练参数指定应该保存模型且本地等级为 0，则将收集到的状态字典保存到指定的输出目录中。
'''
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

'''
作用:进行预训练
输入:sources(),tokenizer(),max_len(最大长度),system_message()
输出:包含已处理对话的输入 ID、目标 ID 和注意力掩码的字典
所选代码是一个用于训练语言模型的较大 Python 脚本的一部分。该代码定义了一个名为 preprocess 的函数，该函数以
对话列表（来源）、分词器和输入序列的最大长度为输入。然后，该函数对输入列表中的每个对话应用一个提示模板，并
返回一个包含已处理对话的输入 ID、目标 ID 和注意力掩码的字典。
提示模板的应用如下：
1.在每个对话的开头添加一个系统消息。
2.将每个对话分割为用户和助手部分。
3.对于每个用户和助手部分，添加一个提示令牌（im_start）在开头，后跟一个表示用户或助手消息的令牌序列。
4.如果对话来自用户，则在用户消息的末尾添加一个 im_end 令牌。如果对话来自助手，则助手消息后跟一个 im_end 令牌。
5.然后将已处理对话的输入 ID、目标 ID 和注意力掩码添加到输出字典中。
输入 ID 用于在训练期间表示输入序列。目标 ID 用于在训练期间表示输出序列。注意力掩码用于屏蔽输入序列中的填充令牌。
preprocess 函数用于为使用 Hugging Face Transformers 库训练语言模型的输入数据进行预处理。然后，
预处理后的数据用作 Hugging Face Transformers 库中 Trainer 类的输入，该类负责训练语言模型。
'''
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

# 监督学习
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    '''
    作用:类初始化，加载和预处理用于训练语言模型的数据
    输入:raw_data, tokenizer, max_len
    所选代码是用于训练语言模型的 Python 脚本的一部分。该代码是一个名为 SupervisedDataset 的自定义 
    PyTorch Dataset 类的 __init__ 方法的一部分。这个类用于加载和预处理用于训练语言模型的数据。
    SupervisedDataset 类的 __init__ 方法接受三个参数：raw_data、tokenizer 和 max_len。raw_data 参数是
    包含训练数据的字典列表。tokenizer 参数是 transformers.PreTrainedTokenizer 类的实例，用于对输入数据进行
    分词。max_len 参数是一个整数，指定输入序列的最大长度。
    在 __init__ 方法内部，代码首先打印一条消息，指示它正在格式化输入。然后，它从 raw_data 列表中的每个字典中
    提取“conversations”字段，并将其存储在名为 sources 的变量中。
    接下来，代码调用 preprocess 函数，传递 sources、tokenizer 和 max_len 参数。preprocess 函数接受这些参数
    ，并返回一个包含输入 ID、标签和注意力掩码的字典，用于输入数据。
    最后，代码将 data_dict 字典中的输入 ID、标签和注意力掩码分别分配给 SupervisedDataset 实例的 input_ids、
    labels 和 attention_mask 属性。
    这段代码是使用 Hugging Face Transformers 库训练语言模型的数据预处理流程的重要部分。它确保输入数据被正确
    分词和格式化，以便在训练过程中使用。
    '''
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

# 惰性监督学习
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

'''
作用:创建用于监督微调的数据集和数据收集器
输入:tokenizer,data_args,max_len
输出:训练和评估数据集的字典
选定的代码是一个名为make_supervised_data_module的函数，负责创建用于监督微调的数据集和数据收集器。它接受三个
参数：tokenizer、data_args和max_len。tokenizer是transformers.PreTrainedTokenizer的一个实例，用于对输入
数据进行标记化。data_args是一个DataArguments的实例，其中包含与数据相关的参数，如数据路径和是否使用惰性预处理。
max_len是一个整数，指定输入序列的最大长度。
该函数首先检查data_args.lazy_preprocess是否为True。如果是，它创建一个LazySupervisedDataset的实例；否则，
创建一个SupervisedDataset的实例。LazySupervisedDataset和SupervisedDataset都是torch.utils.data.Dataset的
自定义子类，后者是用于创建和操作数据集的Python内置库。
然后，该函数使用json.load()从指定的JSON文件中加载训练数据。通过将训练数据、tokenizer和max_len作为参数传递，
创建所选数据集类的实例（LazySupervisedDataset或SupervisedDataset）。
如果指定了data_args.eval_data_path，则从指定的JSON文件加载评估数据，并通过将评估数据、tokenizer和max_len
作为参数传递，创建所选数据集类的实例。如果未指定eval_data_path，则将eval_dataset设置为None。
最后，该函数返回一个包含创建的train_dataset和eval_dataset（如果有的话）的字典。此字典可供训练脚本使用，以访问
训练和评估数据集。
'''
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

'''
作用:训练模型并保存状态
选定的代码是较大的Python脚本的一部分，使用Hugging Face Transformers库训练语言模型。该代码负责训练模型并保存
其状态。
以下是所选代码的详细说明：
1.代码首先通过transformers.HfArgumentParser将命令行参数解析为数据类。此函数允许用户指定各种训练参数，如模型
架构、训练数据集和训练算法。
2.然后，代码检查用户是否指定使用DeepSpeed，这是一个流行的分布式训练库。如果是，则将分布式训练类型设置为
“DeepSpeed”。
3.接着，代码检查用户是否指定使用LoRA（层正则化），一种用于对神经网络的权重矩阵应用低秩近似的技术。如果是，则
设置LoRA的适当参数，如滤波器数量（r）、正则化强度（lora_alpha）和目标模块（lora_target_modules）。
4.代码接着从指定的模型架构加载模型和分词器。它还将填充侧设置为“right”，并禁用快速分词化。
5.如果用户指定使用LoRA，代码使用prepare_model_for_kbit_training函数为量化训练准备模型。此函数启用梯度检查点，
可以提高训练速度。
6.代码接着创建一个LoraConfig对象，用于指定LoRA的参数。如果用户指定使用量化LoRA（QLoRA），则代码设置QLoRA的
适当参数，如每个权重的位数（bits），并禁用指数线性单元（exllama）。
7.接着，代码使用get_peft_model函数将LoRA配置应用于模型。此函数创建一个新模型，将指定的LoRA配置应用于原始模型
的权重矩阵。
8.代码接着打印模型的可训练参数，允许用户验证模型已正确配置。
9.代码然后使用make_supervised_data_module函数加载训练数据。此函数创建一个包含训练数据的Dataset对象，以及
一个可用于填充和截断输入数据的DataCollatorForLanguageModeling对象。
10.代码接着创建一个Trainer对象，负责训练模型。Trainer对象接受模型、分词器、训练参数和训练数据作为输入，并使用
指定的训练算法（如AdamW、Adafactor）来训练模型。
11.代码然后使用Trainer对象的train方法训练模型。此方法将模型训练指定数量的epochs，并将其状态保存到指定的输出
目录。
12.最后，代码使用safe_save_model_for_hf_trainer函数保存模型的状态。此函数以可以被
Hugging Face Transformers库使用的格式保存模型的状态，以及任何指定的偏差参数。
总之，所选代码负责使用Hugging Face Transformers库训练语言模型，并保存其状态以供将来使用。它还包括各种检查和
配置，以确保模型被正确和高效地训练。
'''
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()

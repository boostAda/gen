import os
import argparse
import logging
import torch
import utils
import datetime
import random
import math
import jsonlines
from configparser import ConfigParser
from encode import encoder, encoder_configs
from decode import decoder, decoder_configs


def parse_arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Train or Generate", choices=["T", "G", "E"], default="G")
    parser.add_argument("--config_path", type=str, default="configs/config4.ini")
    parser.add_argument("--gpuid", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/stego")
    #parser.add_argument("--model_dir", type=str, default="THUDM/chatglm2-6b-int4")
    parser.add_argument("--model_dir", type=str, default="gpt2")
    return parser.parse_args()


args = parse_arg_main()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)


def train(args, configs):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    model_type = configs.get("model", "model_type")
    if model_type.lower() == "rnn":
        from models.RNN import RNN
        import trainer
        datafile = configs.get("dataset", "datafile")
        datafile_encoding = configs.get("dataset", "datafile_encoding")
        ctx_len = configs.getint("dataset", "ctx_len")
        is_uncase = configs.getboolean("dataset", "is_uncase")
        word_level = configs.getboolean("dataset", "word_level")
        epoch_length_fixed = configs.getint("trainer", "epoch_length_fixed")
        if not word_level:
            train_dataset = utils.Dataset(open(
                datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed, out_dir, is_uncase,
                                          word_level)
        else:
            min_frequency = configs.getint("vocab", "min_frequency")
            vocab_size = configs.getint("vocab", "vocab_size")
            train_dataset = utils.Dataset(open(
                datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed, out_dir, is_uncase,
                                          word_level, min_frequency, vocab_size)
        model = RNN(
            cell=configs.get("model", "cell"),
            vocab_size=train_dataset.vocab_size,
            embed_size=configs.getint("model", "embed_size"),
            hidden_dim=configs.getint("model", "hidden_dim"),
            num_layers=configs.getint("model", "num_layers"),
            dropout_rate=configs.getfloat("model", "dropout_rate")
        )
        model.to(device)

        tconf = trainer.TrainerConfig(model_type=model_type,
                                      max_epochs=configs.getint("trainer", "max_epochs"),
                                      batch_size=configs.getint("trainer", "batch_size"),
                                      learning_rate=configs.getfloat("trainer", "lr_init"),
                                      lr_decay=True,
                                      lr_final=configs.getfloat("trainer", "lr_final"),
                                      betas=(0.9, 0.99),
                                      eps=configs.getfloat("trainer", "eps"),
                                      grad_norm_clip=configs.getfloat("trainer", "grad_norm_clip"),
                                      warmup_tokens=ctx_len * configs.getint("trainer", "batch_size") * 50,
                                      final_tokens=configs.getint("trainer", "max_epochs") * len(
                                          train_dataset) * ctx_len,
                                      num_workers=configs.getint("trainer", "num_workers"),
                                      epoch_save_frequency=configs.getint("trainer", "epoch_save_frequency"),
                                      epoch_save_path=os.path.join(out_dir,
                                                                   configs.get("trainer", "epoch_save_path")))
        trainer = trainer.Trainer(model, train_dataset, None, tconf)
        trainer.train()
        torch.save(model.state_dict(), os.path.join(out_dir, 'trained-' + configs.get("trainer", "max_epochs") + '-' + trainer.get_run_name()
                                                    +'-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth'))
    elif model_type.lower() == "rwkv":
        from models.RWKV import GPTConfig, GPT
        import trainer
        datafile = configs.get("dataset", "datafile")
        datafile_encoding = configs.get("dataset", "datafile_encoding")
        ctx_len = configs.getint("dataset", "ctx_len")
        is_uncase = configs.getboolean("dataset", "is_uncase")
        word_level = configs.getboolean("dataset", "word_level")
        epoch_length_fixed = configs.getint("trainer", "epoch_length_fixed")
        if not word_level:
            train_dataset = utils.Dataset(open(
                datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed, out_dir, is_uncase,
                                          word_level)
        else:
            min_frequency = configs.getint("vocab", "min_frequency")
            vocab_size = configs.getint("vocab", "vocab_size")
            train_dataset = utils.Dataset(open(
                datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed, out_dir, is_uncase,
                                          word_level, min_frequency, vocab_size)
        model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=configs.get("model", "model_type"),
                              n_layer=configs.getint("model", "n_layer"), n_embd= configs.getint("model", "n_embd"))).to(device)
        tconf = trainer.TrainerConfig(model_type=model_type,
                                      max_epochs=configs.getint("trainer", "max_epochs"),
                                      batch_size=configs.getint("trainer", "batch_size"),
                                      learning_rate=configs.getfloat("trainer", "lr_init"),
                                      lr_decay=True,
                                      lr_final=configs.getfloat("trainer", "lr_final"),
                                      betas=(0.9, 0.99),
                                      eps=configs.getfloat("trainer", "eps"),
                                      grad_norm_clip=configs.getfloat("trainer", "grad_norm_clip"),
                                      warmup_tokens=ctx_len * configs.getint("trainer", "batch_size") * 50,
                                      final_tokens=configs.getint("trainer", "max_epochs") * len(
                                          train_dataset) * ctx_len,
                                      num_workers=configs.getint("trainer", "num_workers"),
                                      epoch_save_frequency=configs.getint("trainer", "epoch_save_frequency"),
                                      epoch_save_path=os.path.join(out_dir,
                                                                   configs.get("trainer", "epoch_save_path")))
        trainer = trainer.Trainer(model, train_dataset, None, tconf)
        trainer.train()
        torch.save(model.state_dict(), os.path.join(out_dir,'trained-' + configs.get("trainer","max_epochs") + '-' + trainer.get_run_name() +
                                                    '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth'))
    elif model_type.lower() == "gpt2-medium":
        from transformers import GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel
        from datasets import load_dataset
        import trainer
        datafile = configs.get("dataset", "datafile")
        model_name_or_path = configs.get("model", "model_name_or_path")
        model_config = GPT2Config.from_pretrained(model_name_or_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=model_config)
        model.to(device)
        extension = "text"
        raw_datasets = load_dataset(extension, data_files=datafile, )

        # special paramters
        RATIO = int(configs.getfloat("gpt2", "split") * 100)
        prompt = configs.get("gpt2", "prompt")
        block_size = configs.getint("dataset", "ctx_len")


        raw_datasets["train"] = load_dataset(extension, data_files=datafile, split=f"train[:{RATIO}%]", )
        raw_datasets["validation"] = load_dataset(extension, data_files=datafile, split=f"train[{RATIO}%:]", )
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def gpt_tokenize_function(examples):
            return tokenizer(tokenizer.bos_token + prompt + examples[text_column_name])
        tokenize_function = gpt_tokenize_function
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=8,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )


        def gpt_group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        group_texts = gpt_group_texts
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=8,
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        trainer.GPT_trainer(lm_datasets, model, configs, tokenizer, out_dir=out_dir, device=device)
    else:
        print("no such model %s".format(args.model))


def generate(args, configs):
    out_dir = args.out_dir
    model_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    model_type = configs.get("model", "model_type")
    ctx_len = configs.getint("dataset", "ctx_len")
    if model_type.lower() in ["rnn", "rwkv"]:
        word_level = configs.getboolean("dataset", "word_level")
        tokenizer = utils.TOKENIZER(os.path.join(out_dir, "vocab"), UNKNOWN_CHAR="\n" if not word_level else "[UNK]")
        if model_type.lower()  == "rnn":
            from models.RNN import RNN
            model = RNN(
                cell=configs.get("model", "cell"),
                vocab_size=tokenizer.vocab_size,
                embed_size=configs.getint("model", "embed_size"),
                hidden_dim=configs.getint("model", "hidden_dim"),
                num_layers=configs.getint("model", "num_layers"),
                dropout_rate=configs.getfloat("model", "dropout_rate")
            )
        elif model_type.lower()  == "rwkv":
            from models.RWKV import GPT, GPTConfig
            model = GPT(
                GPTConfig(tokenizer.vocab_size, ctx_len, model_type=configs.get("model", "model_type"),
                          n_layer=configs.getint("model", "n_layer"), n_embd=configs.getint("model", "n_embd"))).to(
                device)
        model_name = os.path.join(out_dir,configs.get("generate", "model_name"))
        print('loading ' + model_name)
        m2 = torch.load(model_name + '.pth', map_location=device)
        model.load_state_dict(m2)
        model.to(device)
        model.eval()
        with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
        bit_stream = ""
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        max_length = configs.getint("generate", "max_length")
        generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        kwargs = encoder_configs(configs,  alg)
        outfile = "-".join(
            [configs.get("model", "model_tyindicespe"), configs.get("generate", "model_name"), alg, str(generate_num),
             str(max_length)] +
            ["{}{}".format(k, v) for k, v in kwargs.items()])
        print("generation using algorithm: {}".format(alg))
        print("writing into {}".format(os.path.join(out_dir, outfile)))
        for k, v in kwargs.items():
            print("{} : {}".format(k,v))
        with torch.no_grad():
            with jsonlines.open(os.path.join(out_dir, outfile+".jsonl"), "w") as f:
                neg_logits = 0
                stega_idx = 0
                words_num = 0
                bits_num = 0
                sentences = []
                for _ in range(generate_num):
                    neg_logit = 0
                    stega_bits = []
                    if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream += "".join(bit_stream_shuffle)
                    ########################################################################################################
                    context = "[SEP]" if word_level else "\n"
                    stop_word = "[SEP]" if word_level else "\n"
                    ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context.split()] if word_level \
                        else [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
                    src_len = len(ctx)
                    # sample start word
                    pre_ctx = ctx + [0] * (ctx_len - src_len)
                    pre_ctx = [pre_ctx]
                    logits = model.forward(torch.tensor(pre_ctx).cuda())[0][0][0]
                    # logits = torch.nn.functional.log_softmax(logits)
                    # logits -= logits.max()
                    # prob = torch.exp(logits).reshape(-1)
                    prob = logits.softmax(-1)  # fake logits
                    logits = torch.log(prob)  # calculate ppl, this is real logits

                    prob[tokenizer.stoi[stop_word]] = 0
                    prob[tokenizer.UNKNOWN_CHAR] = 0
                    prev = torch.multinomial(prob, 1)
                    neg_logit += -logits[prev]
                    ctx.append(int(prev))
                    sentence = " ".join([tokenizer.itos[i] for i in ctx])
                    # print(words)
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"] # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                    for idx in range(src_len + 1, max_length):
                        pre_ctx = ctx + [0] * (ctx_len - idx)
                        pre_ctx = [pre_ctx]
                        # pre_ctx = [pre_ctx] * 4
                        logits = model.forward(torch.tensor(pre_ctx).cuda())[0][0][idx - 1]
                        # logits = torch.nn.functional.log_softmax(logits)
                        # logits -= logits.max()
                        # prob = torch.exp(logits).reshape(-1)
                        prob = logits.softmax(-1) # fake logits
                        logits = torch.log(prob) # calculate ppl, this is real logits

                        prob[tokenizer.UNKNOWN_CHAR] = 0
                        prob = prob/prob.sum()
                        if alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                        else:
                            prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)

                        # FIXME
                        #  in usual usage,
                        #  if tokenizer.stoi[stop_word] == prev:
                        #      break
                        if tokenizer.stoi[stop_word] == prev or tokenizer.itos[int(prev)] == "[turn]":
                            break
                        ctx.append(int(prev))
                        stega_bits.append(bit_stream[bit_index:bit_index+num_bits_encoded])
                        bit_index += num_bits_encoded
                        neg_logit += -logits[int(prev)]
                        words_num += 1
                        bits_num += num_bits_encoded
                        sentence = " ".join([tokenizer.itos[i] for i in ctx])
                    neg_logits += float(neg_logit)/(idx-1)
                    f.write({"idx": stega_idx,
                             "stego": sentence,
                             "tokens": ctx,
                             "bits": stega_bits})
                    stega_idx += 1
                    sentences.append(sentence.split(context+" ")[1])
                    # print(sentence.strip())
        print("bpw :{:.2f},  model ppl:{:.4f}".format( bits_num/words_num, math.exp(neg_logits/(stega_idx))))
        utils.compute_ppl("gpt2", sentences)
    elif model_type.lower() == "gpt2":
        from transformers import AutoTokenizer, GPT2LMHeadModel
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)

        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable params: {:d}".format(total_trainable_params))
        with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
        bit_stream = ""
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        max_length = configs.getint("generate", "max_length")
        min_length = configs.getint("generate", "min_length")
        #generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        kwargs = encoder_configs(configs, alg)
        outfile = "-".join([configs.get("model","model_type"), configs.get("generate", "model_name"), alg, str(max_length)] +
                           ["{}{}".format(k, v) for k, v in kwargs.items()])
        prompt = configs.get("gpt2", "prompt")
        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_idx = 0
            with jsonlines.open(os.path.join(out_dir, outfile + ".jsonl"), "w") as f,  open('data/writing_prompts/writing_prompts_cover.txt', 'r') as f2:
                for line in f2.readlines():
                    if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream += "".join(bit_stream_shuffle)
                    # sample start word
                    stega_sentence = []
                    # TODO begin
                    #这里的prefix在生成机器文本时可以用前20个token，相当于有前缀的条件生成 也可以不用前缀直接生成，用GPT2在数据集上面微调或者不微调（思考一下不微调行不行），直接生成
                    words = line.strip().split()
                    prefix = ' '.join(words[:20])
                    ##无前缀引导，直接没有数据域知识的盲生成
                    # prefix = ''
                    prompt_text = prompt
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
                                                      add_special_tokens=False,
                                                      return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    stega_bit = []
                    logits = model(input_ids).logits[:, -1, :]
                    logits -= logits.max()
                    probs = torch.exp(logits)
                    for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                        probs[:, forbidden_id] = 0
                    for forbidden_id in range(256):
                        probs[:, forbidden_id] = 0
                    samp = torch.multinomial(probs, 1)
                    stega_sentence.append(int(samp.view(1, 1)))
                    x = torch.cat([input_ids, samp], dim=1)
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"] # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                    for i in range(max_length - 1):
                        if '_EOS' in stega_sentence:
                            break
                        # conditional probability distribution
                        # todo begin
                        log_prob = model(x).logits[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        for forbidden_id in range(256):
                            prob[forbidden_id] = 0
                        # todo end
                        prob = prob / prob.sum()
                        # early stop generation
                        if alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                        else:
                            prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)
                        if int(prev) == tokenizer.eos_token_id and i >= min_length:
                            break
                        stega_sentence.append(int(prev))
                        x = torch.cat([x, prev], dim=1)
                        stega_bit.append(bit_stream[bit_index : bit_index + num_bits_encoded])
                        bit_index += num_bits_encoded

                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)
                    stega_text.append(tokenizer.decode(stega_sentence))
                    f.write({"idx": stega_idx,
                             "stego": tokenizer.decode(stega_sentence),
                             "tokens": stega_sentence,
                             "bits": stega_bit})
    elif model_type.lower() == "chatglm-6b":
        from transformers import AutoTokenizer, AutoModel
        # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable params: {:d}".format(total_trainable_params))
        with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
        bit_stream = ""
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        max_length = configs.getint("generate", "max_length")
        min_length = configs.getint("generate", "min_length")
        #generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        kwargs = encoder_configs(configs, alg)
        outfile = "-".join([configs.get("model","model_type"), configs.get("generate", "model_name"), alg, str(max_length)] +
                           ["{}{}".format(k, v) for k, v in kwargs.items()])
        prompt = configs.get("gpt2", "prompt")
        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_idx = 0
            with jsonlines.open(os.path.join(out_dir, outfile + ".jsonl"), "w") as f,  open('data/aclImdb/cover.txt', 'r') as f2:
                for line in f2.readlines():
                    if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream += "".join(bit_stream_shuffle)
                    # sample start word
                    stega_sentence = []
                    # TODO begin
                    #这里的prefix在生成机器文本时可以用前20个token，相当于有前缀的条件生成 也可以不用前缀直接生成，用GPT2在数据集上面微调或者不微调（思考一下不微调行不行），直接生成
                    words = line.strip().split()
                    prefix = ' '.join(words[:20])
                    prompt_text = prompt
                    # encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
                    #                                   add_special_tokens=False,
                    #                                   return_tensors="pt")
                    encoded_prompt = tokenizer.encode(prefix + prompt_text,
                                                      add_special_tokens=False,
                                                      return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    stega_bit = []
                    logits = model(input_ids).logits[:, -1, :]

                    logits -= logits.max()
                    probs = torch.exp(logits)
                    print(probs)
                    #modification
                    # for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                    #     probs[:, forbidden_id] = 0
                    # for forbidden_id in range(256):
                    #     probs[:, forbidden_id] = 0
                    print("probs=", probs)
                    samp = torch.multinomial(probs, 1)
                    print("samp=", samp)
                    stega_sentence.append(int(samp.view(1, 1)))
                    print(stega_sentence)
                    x = torch.cat([input_ids, samp], dim=1)
                    print("x=", x)
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"] # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                    for i in range(max_length - 1):
                        if '_EOS' in stega_sentence:
                            break
                        # conditional probability distribution
                        # todo begin
                        log_prob = model(x).logits[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        for forbidden_id in range(256):
                            prob[forbidden_id] = 0
                        # todo end
                        prob = prob / prob.sum()
                        # early stop generation
                        if alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                        else:
                            prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)
                        if int(prev) == tokenizer.eos_token_id and i >= min_length:
                            break
                        stega_sentence.append(int(prev))
                        x = torch.cat([x, prev], dim=1)
                        stega_bit.append(bit_stream[bit_index : bit_index + num_bits_encoded])
                        bit_index += num_bits_encoded

                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)
                    stega_text.append(tokenizer.decode(stega_sentence))
                    f.write({"idx": stega_idx,
                             "stego": tokenizer.decode(stega_sentence),
                             "tokens": stega_sentence,
                             "bits": stega_bit})
    elif model_type.lower() == "opt-2.7b":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable params: {:d}".format(total_trainable_params))
        with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
            bit_stream_ori = f.read().strip()
        bit_stream = list(bit_stream_ori)
        bit_stream = ''.join(bit_stream)
        bit_stream = ""
        bit_index = int(torch.randint(0, high=10000, size=(1,)))
        max_length = configs.getint("generate", "max_length")
        min_length = configs.getint("generate", "min_length")
        #generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        kwargs = encoder_configs(configs, alg)
        outfile = "-".join([configs.get("model","model_type"), configs.get("generate", "model_name"), alg, str(max_length)] +
                           ["{}{}".format(k, v) for k, v in kwargs.items()])
        prompt = configs.get("gpt2", "prompt")
        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_idx = 0
            with jsonlines.open(os.path.join(out_dir, outfile + ".jsonl"), "w") as f,  open('data/aclImdb/cover.txt', 'r') as f2:
                for line in f2.readlines():
                    if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
                        bit_stream_shuffle = list(bit_stream_ori)
                        random.shuffle(bit_stream_shuffle)
                        bit_stream += "".join(bit_stream_shuffle)
                    # sample start word
                    stega_sentence = []
                    # TODO begin
                    #这里的prefix在生成机器文本时可以用前20个token，相当于有前缀的条件生成 也可以不用前缀直接生成，用GPT2在数据集上面微调或者不微调（思考一下不微调行不行），直接生成
                    words = line.strip().split()
                    prefix = ' '.join(words[:20])
                    prompt_text = prompt
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
                                                      add_special_tokens=False,
                                                      return_tensors="pt")
                    # encoded_prompt = tokenizer.encode(prefix + prompt_text,
                    #                                   add_special_tokens=False,
                    #                                   return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    stega_bit = []
                    logits = model(input_ids).logits[:, -1, :]

                    logits -= logits.max()
                    probs = torch.exp(logits)
                    #modification
                    for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                        probs[:, forbidden_id] = 0
                    for forbidden_id in range(256):
                        probs[:, forbidden_id] = 0
                    samp = torch.multinomial(probs, 1)
                    stega_sentence.append(int(samp.view(1, 1)))
                    x = torch.cat([input_ids, samp], dim=1)
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"] # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                    for i in range(max_length - 1):
                        if '_EOS' in stega_sentence:
                            break
                        # conditional probability distribution
                        # todo begin
                        log_prob = model(x).logits[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        for forbidden_id in range(256):
                            prob[forbidden_id] = 0
                        # todo end
                        prob = prob / prob.sum()
                        # early stop generation
                        if alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
                        else:
                            prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)
                        if int(prev) == tokenizer.eos_token_id and i >= min_length:
                            break
                        stega_sentence.append(int(prev))
                        x = torch.cat([x, prev], dim=1)
                        stega_bit.append(bit_stream[bit_index : bit_index + num_bits_encoded])
                        bit_index += num_bits_encoded

                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)
                    stega_text.append(tokenizer.decode(stega_sentence))
                    f.write({"idx": stega_idx,
                             "stego": tokenizer.decode(stega_sentence),
                             "tokens": stega_sentence,
                             "bits": stega_bit})
    # elif model_type.lower() == "gpt2":
    #     from transformers import AutoTokenizer, GPT2LMHeadModel
    #     tokenizer = AutoTokenizer.from_pretrained(out_dir)
    #     model = GPT2LMHeadModel.from_pretrained(out_dir)
    #     # tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=False)
    #     # model = AutoModel.from_pretrained("chatglm-6b", trust_remote_code=False).half().cuda()
    #     model.to(device)
    #     total_params = sum(p.numel() for p in model.parameters())
    #     print("Total params: {:d}".format(total_params))
    #     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("Trainable params: {:d}".format(total_trainable_params))
    #     with open(configs.get("generate", "bit_filepath"), 'r', encoding='utf8') as f:
    #         bit_stream_ori = f.read().strip()
    #     bit_stream = list(bit_stream_ori)
    #     bit_stream = ''.join(bit_stream)
    #     bit_stream = ""
    #     bit_index = int(torch.randint(0, high=10000, size=(1,)))
    #     max_length = configs.getint("generate", "max_length")
    #     generate_num = configs.getint("generate", "generate_num")
    #     alg = configs.get("generate", "alg")
    #     kwargs = encoder_configs(configs, alg)
    #     outfile = "-".join([configs.get("model","model_type"), configs.get("generate", "model_name"), alg, str(generate_num), str(max_length)] +
    #                        ["{}{}".format(k, v) for k, v in kwargs.items()])
    #     prompt = configs.get("gpt2", "prompt")
    #     model.eval()
    #     with torch.no_grad():
    #         stega_text = []
    #         stega_idx = 0
    #         with jsonlines.open(os.path.join(out_dir, outfile + ".jsonl"), "w") as f:
    #             while len(stega_text) < generate_num:
    #                 if len(bit_stream) <= max_length * math.log2(tokenizer.vocab_size):
    #                     bit_stream_shuffle = list(bit_stream_ori)
    #                     random.shuffle(bit_stream_shuffle)
    #                     bit_stream += "".join(bit_stream_shuffle)
    #                 # sample start word
    #                 stega_sentence = []
    #                 # TODO begin
    #                 prefix = ""
    #                 prompt_text = prompt
    #                 encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text,
    #                                                   add_special_tokens=False,
    #                                                   return_tensors="pt")
    #                 encoded_prompt = encoded_prompt.to(device)
    #                 input_ids = encoded_prompt
    #                 stega_bit = []
    #                 logits = model(input_ids).logits[:, -1, :]
    #                 logits -= logits.max()
    #                 probs = torch.exp(logits)
    #                 for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
    #                     probs[:, forbidden_id] = 0
    #                 for forbidden_id in range(256):
    #                     probs[:, forbidden_id] = 0
    #                 samp = torch.multinomial(probs, 1)
    #                 stega_sentence.append(int(samp.view(1, 1)))
    #                 x = torch.cat([input_ids, samp], dim=1)
    #                 if alg.lower() == "ac":
    #                     max_val = 2 ** kwargs["precision"] # num of intervals
    #                     # max_val = 2**52
    #                     cur_interval = [0, max_val]
    #                 for i in range(max_length - 1):
    #                     if '_EOS' in stega_sentence:
    #                         break
    #                     # conditional probability distribution
    #                     # todo begin
    #                     log_prob = model(x).logits[:, -1, :]
    #                     log_prob -= log_prob.max()
    #                     prob = torch.exp(log_prob).reshape(-1)
    #                     for forbidden_id in range(256):
    #                         prob[forbidden_id] = 0
    #                     # todo end
    #                     prob = prob / prob.sum()
    #                     # early stop generation
    #                     if alg.lower() == "ac":
    #                         cur_interval, prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, cur_interval, **kwargs)
    #                     else:
    #                         prev, num_bits_encoded = encoder(alg, prob, bit_stream, bit_index, **kwargs)
    #                     if int(prev) == tokenizer.eos_token_id:
    #                         break
    #                     stega_sentence.append(int(prev))
    #                     x = torch.cat([x, prev], dim=1)
    #                     bit_index += num_bits_encoded
    #
    #                 if tokenizer.eos_token_id in stega_sentence:
    #                     stega_sentence.remove(tokenizer.eos_token_id)
    #                 stega_text.append(tokenizer.decode(stega_sentence))
    #                 f.write({"idx": stega_idx,
    #                          "stego": tokenizer.decode(stega_sentence),
    #                          "tokens": stega_sentence,
    #                          "bits": stega_bit})
    else:
        print("no such model %s".format(args.model))


def extract(args, configs):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    model_type = configs.get("model", "model_type")
    ctx_len = configs.getint("dataset", "ctx_len")
    infile = configs.get("extract", "in_filepath")
    if model_type.lower() in ["rnn", "rwkv"]:
        word_level = configs.getboolean("dataset", "word_level")
        tokenizer = utils.TOKENIZER(os.path.join(out_dir, "vocab"), UNKNOWN_CHAR="\n" if not word_level else "[UNK]")
        if model_type.lower() == "rnn":
            from models.RNN import RNN
            model = RNN(
                cell=configs.get("model", "cell"),
                vocab_size=tokenizer.vocab_size,
                embed_size=configs.getint("model", "embed_size"),
                hidden_dim=configs.getint("model", "hidden_dim"),
                num_layers=configs.getint("model", "num_layers"),
                dropout_rate=configs.getfloat("model", "dropout_rate")
            )
        elif model_type.lower() == "rwkv":
            from models.RWKV import GPT, GPTConfig
            model = GPT(
                GPTConfig(tokenizer.vocab_size, ctx_len, model_type=configs.get("model", "model_type"),
                          n_layer=configs.getint("model", "n_layer"), n_embd=configs.getint("model", "n_embd"))).to(
                device)
        model_name = os.path.join(out_dir, configs.get("generate", "model_name"))
        print('loading ' + model_name)
        m2 = torch.load(model_name + '.pth', map_location=device)
        model.load_state_dict(m2)
        model.to(device)
        model.eval()
        max_length = configs.getint("generate", "max_length")
        generate_num = configs.getint("generate", "generate_num")
        alg = configs.get("generate", "alg")
        kwargs = decoder_configs(configs, alg)
        outfile = "-".join(
            ["decode", configs.get("model", "model_type"), configs.get("generate", "model_name"), alg, str(generate_num),
             str(max_length)] +
            ["{}{}".format(k, v) for k, v in kwargs.items()])
        print("generation using algorithm: {}".format(alg))
        print("writing into {}".format(os.path.join(out_dir, outfile)))
        for k, v in kwargs.items():
            print("{} : {}".format(k, v))
        with torch.no_grad():
            with open(infile, "r") as f_in, jsonlines.open(os.path.join(out_dir, outfile + ".jsonl"), "w") as f_out:
                stega_idx = 0
                while True:
                    sentence = f_in.readline()
                    if not sentence:
                        print("Finished Extraction")
                        break
                    if sentence == "\n":
                        continue
                    if not word_level:
                        sentence = "\n"+"".join(sentence.split())
                    else:
                        if "[SEP]" in sentence:
                            sentence = "[SEP]" + "".join(sentence.split("[SEP]")[1])
                        else:
                            sentence = "[SEP] " + sentence
                    stega_bits = []
                    ########################################################
                    context = sentence
                    token_ids = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context.split()] if word_level \
                        else [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
                    L = len(token_ids)
                    if L > ctx_len:
                        continue
                    token_ids += [0] * (ctx_len - L)
                    log_probs = model(torch.LongTensor([token_ids]).to(device))[0]
                    if alg.lower() == "ac":
                        max_val = 2 ** kwargs["precision"] # num of intervals
                        # max_val = 2**52
                        cur_interval = [0, max_val]
                    for curr in range(1, L - 1):
                        log_prob = log_probs[0, curr, :]
                        # log_prob -= log_prob.max()
                        # prob = torch.exp(log_prob).reshape(-1)
                        # prob = prob / prob.sum()
                        prob = log_prob.softmax(-1)
                        prob[tokenizer.UNKNOWN_CHAR] = 0
                        prob = prob/prob.sum()
                        if token_ids[curr + 1] == tokenizer.UNKNOWN_CHAR:
                            stega_bits.append("")
                            continue
                        else:
                            if alg.lower() == "ac":
                                cur_interval, stega_bits_tmp = decoder(alg, prob, token_ids[curr + 1], cur_interval,
                                                                       **kwargs)
                                stega_bits.append(stega_bits_tmp)
                            else:
                                stega_bits.append(decoder(alg, prob, token_ids[curr + 1], **kwargs))
                    f_out.write({"stego": sentence,
                                 "tokens": token_ids[:L],
                                 "idx": stega_idx, "bits": stega_bits})
                    stega_idx += 1
        ############################################################################################################
        # check decoder
        # check_file = "-".join(
        #             [configs.get("model", "model_type"), configs.get("generate", "model_name"), alg, str(generate_num),
        #              str(max_length)] +
        #             ["{}{}".format(k, v) for k, v in kwargs.items()])
        # check_bits = []
        # with open(os.path.join(out_dir, check_file+".jsonl"), "r") as f_in :
        #     for item in jsonlines.Reader(f_in):
        #         check_bits.append("".join(item["bits"]))
        # bits = []
        # with open(os.path.join(out_dir, outfile+".jsonl"), "r") as f_in :
        #     for item in jsonlines.Reader(f_in):
        #         bits.append("".join(item["bits"]))
        # error = 0
        # id = 0
        # for bit, check_bit in zip(bits, check_bits):
        #     if bit != check_bit:
        #         print("error id :{}".format(id))
        #         error += 1
        #     id += 1
        # print("error : {}".format(error) )


# def jsonl2txt(infilepath, outfilepath):
#     with open(infilepath, "r") as f_in, open(outfilepath, "w") as f_out:
#         for item in jsonlines.Reader(f_in):
#             f_out.write(item["stego"] + "\n")
# infile = "RWKV-trained-1-ac-100-128-precision52"
# infile_path = os.path.join("out", infile+".jsonl")
# outfile_path = os.path.join("out", infile+".txt")
# jsonl2txt(infile_path, outfile_path)


if __name__ == '__main__':
    configs = ConfigParser()
    configs.read(args.config_path)
    utils.set_seed(args.seed)
    if args.mode == "T":
        train(args, configs)
    elif args.mode == "G":
        generate(args, configs)
    elif args.mode == "E":
        extract(args, configs)
    else:
        print("no such mode %s".format(args.mode))

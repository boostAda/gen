from torch.utils.data.dataloader import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
import logging
import os
import datetime
import sys
import math


logger = logging.getLogger(__name__)
log_file = open("mylog.txt", "a")


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    lr_decay = True  # linear warmup followed by cosine decay
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0
        self.device = 'cpu'
        if torch.cuda.is_available():  # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()

    def get_run_name(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + \
            cfg.model_type + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            if config.num_workers > 0:
                loader = DataLoader(data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            else:
                loader = DataLoader(data, shuffle=False,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(
                loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)

            for it, (x, y) in pbar:
                x = x.to(self.device)  # place data on the correct device
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    _, loss = model(x, y)  # forward the model

                if is_train:  # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()

                    if config.grad_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm_clip)

                    optimizer.step()

                    if config.lr_decay:  # decay the learning rate based on our progress
                        # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += (y >= 0).sum()
                        lr_final_factor = config.lr_final / config.learning_rate
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = lr_final_factor + \
                                (1 - lr_final_factor) * float(self.tokens) / \
                                float(config.warmup_tokens)
                            progress = 0
                        else:
                            # exponential learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            if progress >= 1:
                                lr_mult = lr_final_factor
                            else:
                                lr_mult = math.exp(math.log(lr_final_factor) * pow(progress, 1))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    now_loss = loss.item()  # report progress
                    self.lr = lr

                    self.steps += 1

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * \
                            (1.0 - factor) + now_loss * factor
                    pbar.set_description(
                        f"mini-epoch {epoch+1} prog {progress*100.0:.2f}% iter {it}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {lr:e}")

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')

            log_file.write(
                f'{epoch+1} {self.avg_loss:.6f} {math.exp(self.avg_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} \n')
            log_file.flush()

            if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (epoch == config.max_epochs - 1):
                # DataParallel wrappers keep raw model object in .module
                raw_model = self.model.module if hasattr(
                    self.model, "module") else self.model
                torch.save(raw_model.state_dict(),
                           self.config.epoch_save_path + str(epoch+1) + '.pth')


def GPT_trainer(dataset, model, configs, tokenizer, out_dir ,device):
    from transformers import set_seed, default_data_collator, AdamW, get_scheduler
    def gpt_eval(dataset, model, configs, tokenizer, device):
        eval_dataset = dataset["validation"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=default_data_collator, batch_size=configs.getint("trainer", "batch_size")
        )
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: batch[k].to(device) for k in batch}
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.repeat(configs.getint("trainer", "batch_size")))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        return perplexity

    train_dataset = dataset["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=configs.getint("trainer", "batch_size")
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": configs.getfloat("gpt2", "weight_decay")
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.getfloat("trainer", "lr_init"))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    max_epochs = math.ceil(configs.getint("trainer", "max_epochs") * configs.getint("trainer", "epoch_length_fixed") / num_update_steps_per_epoch)
    max_train_steps = int(max_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=configs.get("gpt2", "lr_scheduler_type"),
        optimizer=optimizer,
        num_warmup_steps=int(configs.getfloat("gpt2", "warmup_ratio") * max_train_steps),
        num_training_steps=max_train_steps,
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_update_steps_per_epoch}")
    logger.info(f"  Num Epochs = {max_epochs}")
    logger.info("  Instantaneous batch size per device = {}".format(configs.getint("trainer", "batch_size")))
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=False)
    completed_steps = 0
    best_eavl_ppl = 100000000

    for epoch in range(max_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: batch[k].to(device) for k in batch}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps % configs.getint("gpt2", "GENERATE_EVERY") == 0:
                model.eval()
                logger.info(tokenizer.decode(model.generate(do_sample=True)[0]))
                model.train()

            if completed_steps % configs.getint("gpt2", "EVAL_STEPS") == 0:
                perplexity = gpt_eval(dataset, model, configs, tokenizer, device)
                logger.info(f"global step {completed_steps}: perplexity: {perplexity}")
                if perplexity < best_eavl_ppl:
                    best_eavl_ppl = perplexity
                    unwrapped_model = model
                    unwrapped_model.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)

                model.train()

            if completed_steps >= max_train_steps:
                break

    unwrapped_model = model
    unwrapped_model.save_pretrained(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    return model


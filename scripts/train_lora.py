#!/usr/bin/env python
"""Reproducible real-data LoRA training on common owned windows.

Bounded pilots by default. --max-steps 0 trains one pass over every window;
--checkpoint-every/--resume make long runs restartable; --thoughts selects the
self_thought_v3 causal recipe (the base model's own saved closed thought as
unscored context before the supervised final JSON).
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.modeling import ChapterModel
from chaptering.supervision import SupervisionError, thought_prefix_from_raw
from pilot_seq2seq import gpu0_lock


def attach_thoughts(items, results_path, alias):
    """Join saved zero-shot generations to training windows by video/window key.

    Returns (usable items with ``thought_prefix``, drop counts). A window is dropped,
    never patched, when its generation is missing, did not stop normally, or has no
    single closed native thought.
    """
    saved = {}
    for line in results_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            saved[(row["video_id"], row["window_index"])] = row
    usable, dropped = [], defaultdict(int)
    for item in items:
        row = saved.get((item["window"]["video_id"], item["window"]["window_index"]))
        if row is None:
            dropped["no_saved_generation"] += 1
        elif row.get("finish_reason") != "stop":
            dropped["generation_not_finished"] += 1
        else:
            prefix = thought_prefix_from_raw(row.get("raw_text"), alias)
            if prefix is None:
                dropped["no_closed_native_thought"] += 1
            else:
                usable.append(dict(item, thought_prefix=prefix))
    return usable, dict(dropped)


def training_order(items, seed):
    """Round-robin videos, so a step cap cannot consume only the longest video."""
    groups = defaultdict(list)
    for item in items:
        groups[item["window"]["video_id"]].append(item)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    depth = 0
    while any(depth < len(group) for group in groups.values()):
        for key in keys:
            if depth < len(groups[key]):
                result.append(groups[key][depth])
        depth += 1
    return result


def estimated_tokens(item, alias):
    """Cheap length estimate for batching only; exact counts are logged per step."""
    return item["window"]["input_token_counts"][alias] + len(item.get("thought_prefix", "")) // 3 + 256


def plan_token_batches(order, alias, batch_tokens, seed, pool=64, max_rows=0):
    """Group the training order into padded batches under a token budget.

    ``batch_tokens`` 0 keeps one window per step in the given order. Otherwise
    consecutive pools of windows are sorted by length, packed so that
    ``windows * longest_estimate <= batch_tokens`` (always at least one window,
    at most ``max_rows`` windows when that is nonzero),
    and the resulting batches are shuffled with the run seed.
    """
    if not batch_tokens:
        return [[item] for item in order]
    batches = []
    for start in range(0, len(order), pool):
        group = sorted(order[start:start + pool], key=lambda item: -estimated_tokens(item, alias))
        current = []
        for item in group:
            longest = estimated_tokens(current[0] if current else item, alias)
            if current and ((len(current) + 1) * longest > batch_tokens or (max_rows and len(current) == max_rows)):
                batches.append(current)
                current = []
            current.append(item)
        if current:
            batches.append(current)
    random.Random(seed).shuffle(batches)
    return batches


def learning_rate_at(step, total, base, warmup, schedule):
    """Learning rate for 0-based optimizer step ``step`` of ``total``."""
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if schedule == "cosine" and total > warmup:
        return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    return base


def collate(torch, prepared, pad_id, seq2seq):
    """Right-pad single-example tensors into one batch; padding gets no attention or loss."""
    def pad(name, value):
        rows = [inputs[name][0] for inputs in prepared]
        width = max(row.shape[0] for row in rows)
        out = torch.full((len(rows), width), value, dtype=torch.long, device=rows[0].device)
        for index, row in enumerate(rows):
            out[index, :row.shape[0]] = row
        return out
    batch = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0)}
    if seq2seq:
        batch.update(labels=pad("labels", -100), decoder_input_ids=pad("decoder_input_ids", pad_id),
                     decoder_attention_mask=pad("decoder_attention_mask", 0))
    else:
        batch["labels"] = pad("labels", -100)
    return batch


def supervised_tail(labels):
    """Shortest common tail of a right-padded causal batch that covers every supervised next token.

    Returns ``keep`` such that logits for the last ``keep`` input positions predict every label
    that is not -100 (the logit at position t predicts label t + 1).
    """
    columns = (labels[:, 1:] != -100).any(dim=0).nonzero()
    if not len(columns):
        raise ValueError("Batch has no supervised token")
    return labels.shape[1] - int(columns[0])


def selective_causal_loss(torch, model, inputs):
    """Mean next-token cross entropy over supervised tokens, computing logits only for the
    supervised tail. Matches the full-logit Hugging Face causal loss (float upcast, mean over
    non-ignored tokens) while avoiding the [batch, sequence, vocabulary] logit tensor."""
    labels = inputs["labels"]
    keep = supervised_tail(labels)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   logits_to_keep=keep).logits
    if logits.shape[1] != keep:
        raise RuntimeError("Model ignored logits_to_keep")
    targets = labels[:, labels.shape[1] - keep + 1:]
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             targets.reshape(-1), ignore_index=-100)


def prepare_rft(runtime, item, torch):
    """rft_self_samples_v1: the base model's own sampled completion (native thought + final answer),
    selected on training windows by agreement with gold, supervised in full after the exact generation
    prompt, plus the native turn terminator. No text is written or edited by us."""
    from chaptering.windowing import build_window_prompt
    tokenizer = runtime.tokenizer
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": build_window_prompt(item["window"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    terminal = tokenizer.convert_tokens_to_ids("<|im_end|>" if runtime.alias == "M1" else "<turn|>")
    completion = tokenizer.encode(item["rft_raw"], add_special_tokens=False) + [terminal]
    ids = prompt + completion
    if len(ids) > runtime.decoder_limit:
        raise ValueError("RFT sequence exceeds pinned context")
    inputs = {"input_ids": torch.tensor([ids], device="cuda:0"),
              "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda:0"),
              "labels": torch.tensor([[-100] * len(prompt) + completion], device="cuda:0")}
    return inputs, {"recipe": "rft_self_samples_v1", "prompt_tokens": len(prompt),
                    "supervised_tokens": len(completion), "supervision_gate_passed": True,
                    "rft_meta": item.get("rft_meta")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", choices=["E1", "E2", "M1", "M2"], required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=393)
    parser.add_argument("--gemma-final-context", choices=["direct_v1", "closed_thought_v2"], default="direct_v1")
    parser.add_argument("--thoughts", type=Path,
                        help="saved zero-shot windows.jsonl for these training windows (self_thought_v3)")
    parser.add_argument("--max-sequence-tokens", type=int,
                        help="skip (and count) windows whose full training sequence is longer")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="windows between checkpoints; 0 disables")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run in --output")
    parser.add_argument("--batch-tokens", type=int, default=0,
                        help="padded-batch budget (windows * longest window tokens); 0 = one window per step")
    parser.add_argument("--max-batch-windows", type=int, default=0,
                        help="cap on windows per padded batch (bounds the selective-logit tail); 0 = no cap")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--selective-logits", action="store_true",
                        help="causal only: logits for the supervised tail only (same loss, far less memory)")
    args = parser.parse_args()
    if args.selective_logits and args.alias not in {"M1", "M2"}:
        parser.error("--selective-logits applies to causal models only")
    if args.max_steps < 0 or args.rank < 1 or args.learning_rate <= 0 or args.grad_accum < 1 or args.checkpoint_every < 0:
        raise ValueError("Nonnegative steps and positive rank, learning rate and accumulation required")
    if args.alias != "M2" and args.gemma_final_context != "direct_v1":
        parser.error("Gemma context override requires alias M2")
    if args.thoughts and (args.alias not in {"M1", "M2"} or args.gemma_final_context != "direct_v1"):
        parser.error("--thoughts is only for causal models without another final context")
    if args.resume:
        if not (args.output / "checkpoint/state.json").is_file():
            parser.error("--resume needs an existing checkpoint in --output")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    items = [json.loads(line) for line in args.windows.read_text().splitlines() if line.strip()]
    if not items or any(set(item) - {"rft_raw", "rft_meta"} != {"window", "target"} for item in items):
        raise ValueError("Training requires explicit window/target records")
    manifest_path = args.windows.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("preparation_gate_passed", manifest.get("passed", False)):
        raise ValueError("Window preparation gate must pass before training")
    for name in ["train_windows.jsonl", "train_records.jsonl", "validation_records.jsonl"]:
        digest = hashlib.sha256((args.windows.parent / name).read_bytes()).hexdigest()
        if manifest.get("files_sha256", {}).get(name) != digest:
            raise ValueError("Prepared window/record hash mismatch: " + name)
    # Verify selected training IDs from the canonical training file, not filenames alone.
    training_ids = {json.loads(line)["video_id"] for line in (args.windows.parent / "train_records.jsonl").read_text().splitlines()}
    validation_ids = {json.loads(line)["video_id"] for line in (args.windows.parent / "validation_records.jsonl").read_text().splitlines()}
    if training_ids & validation_ids or any(item["window"]["video_id"] not in training_ids for item in items):
        raise ValueError("Training/validation manifest isolation failed")
    available_windows, thought_drops = len(items), None
    if args.thoughts:
        items, thought_drops = attach_thoughts(items, args.thoughts, args.alias)
        if not items:
            raise ValueError("No training window has a usable saved thought")
    order = training_order(items, args.seed)
    if args.max_steps:
        order = order[:args.max_steps]
    batches = plan_token_batches(order, args.alias, args.batch_tokens, args.seed, max_rows=args.max_batch_windows)
    optimizer_steps_planned = math.ceil(len(batches) / args.grad_accum)
    report = {"alias": args.alias, "status": "running", "batch_tokens": args.batch_tokens,
              "max_batch_windows": args.max_batch_windows,
              "planned_batches": len(batches), "planned_optimizer_steps": optimizer_steps_planned,
              "gradient_checkpointing": not args.no_gradient_checkpointing,
              "target_modules": args.target_modules.split(","), "warmup_steps": args.warmup_steps,
              "lr_schedule": args.lr_schedule, "selective_logits": args.selective_logits,
              "purpose": "bounded real-data learning pilot" if args.max_steps else "one pass over all prepared windows",
              "final_context": ("rft_self_samples_v1" if any("rft_raw" in item for item in items) else
                                "self_thought_v3" if args.thoughts else
                                args.gemma_final_context if args.alias == "M2" else "native_template_v1"),
              "thoughts_sha256": hashlib.sha256(args.thoughts.read_bytes()).hexdigest() if args.thoughts else None,
              "windows_dropped_without_usable_thought": thought_drops,
              "max_sequence_tokens": args.max_sequence_tokens, "grad_accum": args.grad_accum,
              "checkpoint_every": args.checkpoint_every,
              "available_windows": available_windows, "planned_steps": len(order), "available_training_videos": len(training_ids),
              "planned_training_videos": len({item["window"]["video_id"] for item in order}),
              "rank": args.rank, "lora_alpha": 2 * args.rank, "lora_dropout": 0.05,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "gemma_final_context": args.gemma_final_context if args.alias == "M2" else None,
              "windows_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
              "packages": {name: importlib.metadata.version(name)
                           for name in ["torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"]}}
    if args.resume:
        previous = json.loads((args.output / "run_config.json").read_text())
        if any(previous.get(key) != report[key] for key in report if key != "packages"):
            raise ValueError("Resume settings differ from the interrupted run")
    else:
        (args.output / "run_config.json").write_text(json.dumps(report, indent=2))
    started = time.perf_counter()
    with gpu0_lock():
        import torch
        from peft import (LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, PeftModel,
                          set_peft_model_state_dict)
        torch.manual_seed(args.seed)
        runtime = ChapterModel(root, args.alias, attention=args.attention,
                               gemma_final_context=args.gemma_final_context)
        report.update(revision=runtime.entry["revision"], attention=runtime.attention,
                      loaded_seconds=time.perf_counter() - started)
        task = TaskType.SEQ_2_SEQ_LM if runtime.seq2seq else TaskType.CAUSAL_LM
        runtime.model = get_peft_model(runtime.model, LoraConfig(task_type=task, r=args.rank,
            lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=args.target_modules.split(",")))
        if not args.no_gradient_checkpointing:
            runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        runtime.model.config.use_cache = False
        if hasattr(runtime.model.config, "text_config"):
            runtime.model.config.text_config.use_cache = False
        runtime.model.train()
        trainable = [(name, p) for name, p in runtime.model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate)
        report["trainable_parameters"] = sum(p.numel() for _, p in trainable)
        report["adapter_modules"] = [name for name, module in runtime.model.named_modules() if hasattr(module, "lora_A")]
        report["exposed_videos"] = []
        losses, step_times, token_total = [], [], 0
        any_update, first_index, skipped = False, 0, defaultdict(int)
        optimizer_step = 0
        pad_id = runtime.tokenizer.pad_token_id if runtime.tokenizer.pad_token_id is not None else 0
        checkpoint_dir = args.output / "checkpoint"
        if args.resume:
            from safetensors.torch import load_file
            state = json.loads((checkpoint_dir / "state.json").read_text())
            set_peft_model_state_dict(runtime.model, load_file(str(checkpoint_dir / "adapter/adapter_model.safetensors")))
            optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cuda:0"))
            first_index, any_update = state["next_index"], True
            optimizer_step = state["optimizer_steps"]
            skipped.update(state["skipped"])
            kept = [json.loads(line) for line in (args.output / "steps.jsonl").read_text().splitlines()][:state["logged_steps"]]
            (args.output / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept))
            losses = [row["loss"] for row in kept]
            step_times = [row["seconds"] for row in kept]
            token_total = sum(row["teacher_forcing_tokens"] for row in kept)
            report["exposed_videos"] = [video for row in kept for video, _ in row.get("windows", [[row["video_id"], 0]])]
            report["resumed_from_index"] = first_index

        def save_checkpoint(next_index):
            temporary = args.output / "checkpoint.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            runtime.model.save_pretrained(temporary / "adapter")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            (temporary / "state.json").write_text(json.dumps(
                {"next_index": next_index, "logged_steps": len(losses), "skipped": dict(skipped),
                 "optimizer_steps": optimizer_step}))
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            temporary.replace(checkpoint_dir)

        optimizer.zero_grad(set_to_none=True)
        pending = 0
        with (args.output / "steps.jsonl").open("a") as log:
            for index in range(first_index, len(batches)):
                phase = time.perf_counter()
                prepared, audits, used = [], [], []
                for item in batches[index]:
                    try:
                        single, audit = (prepare_rft(runtime, item, torch) if "rft_raw" in item
                                         else runtime.prepare_training(item))
                    except SupervisionError:
                        if not args.thoughts:
                            raise
                        skipped["self_thought_supervision_rejected"] += 1
                        continue
                    length = single["input_ids"].shape[-1]
                    if not runtime.seq2seq and length > runtime.decoder_limit:
                        raise ValueError("Full causal teacher-forcing sequence exceeds pinned context")
                    if args.max_sequence_tokens and length > args.max_sequence_tokens:
                        skipped["longer_than_max_sequence_tokens"] += 1
                        continue
                    prepared.append(single)
                    audits.append(audit)
                    used.append(item)
                last = index == len(batches) - 1
                if not prepared:
                    if not (last and pending):
                        continue
                    inputs = loss = None
                else:
                    inputs = prepared[0] if len(prepared) == 1 else collate(torch, prepared, pad_id, runtime.seq2seq)
                    if args.selective_logits and "selective_logits_check" not in report:
                        # Same loss either way: verify once on this batch's first window, dropout off.
                        runtime.model.eval()
                        with torch.no_grad():
                            full = float(runtime.model(**prepared[0]).loss)
                            tail = float(selective_causal_loss(torch, runtime.model, prepared[0]))
                        runtime.model.train()
                        report["selective_logits_check"] = {"full_logit_loss": full, "selective_loss": tail}
                        if not math.isclose(full, tail, rel_tol=2e-3, abs_tol=2e-3):
                            raise RuntimeError("Selective-logit loss differs from the full-logit loss")
                    if args.selective_logits:
                        loss = selective_causal_loss(torch, runtime.model, inputs)
                    else:
                        loss = runtime.model(**inputs).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    (loss / args.grad_accum).backward()
                    pending += 1
                if pending == args.grad_accum or (last and pending):
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in trainable):
                        raise RuntimeError("Nonfinite adapter gradient")
                    nonzero = [(name, p) for name, p in trainable if p.grad is not None and torch.count_nonzero(p.grad).item()]
                    if not nonzero:
                        raise RuntimeError("No nonzero adapter gradient")
                    changed_parameter = nonzero[0][1]
                    before = changed_parameter.detach().clone() if not any_update else None
                    torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate_at(optimizer_step, optimizer_steps_planned, args.learning_rate,
                                                       args.warmup_steps, args.lr_schedule)
                    optimizer.step()
                    optimizer_step += 1
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    if any(not torch.isfinite(p).all() for _, p in trainable):
                        raise RuntimeError("Nonfinite adapter weight")
                    if before is not None:
                        any_update = bool(torch.count_nonzero(before != changed_parameter).item())
                        if not any_update:
                            raise RuntimeError("First optimizer step made no adapter update")
                torch.cuda.synchronize()
                if loss is None:
                    continue
                elapsed = time.perf_counter() - phase
                loss_value = float(loss.detach())
                losses.append(loss_value)
                step_times.append(elapsed)
                count = sum(single["input_ids"].shape[-1] + (single["labels"].shape[-1] if runtime.seq2seq else 0)
                            for single in prepared)
                token_total += count
                row = {"step": len(losses), "windows": [[item["window"]["video_id"], item["window"]["window_index"]] for item in used],
                       "video_id": used[0]["window"]["video_id"], "window_index": used[0]["window"]["window_index"],
                       "loss": loss_value, "seconds": elapsed, "teacher_forcing_tokens": count,
                       "padded_tokens": int(inputs["input_ids"].numel()), "optimizer_steps": optimizer_step,
                       "supervised_tokens": sum(audit.get("supervised_tokens", 0) for audit in audits),
                       "supervision_audit": audits[0],
                       "title_supervision_stats": used[0]["target"]["title_supervision_stats"]}
                log.write(json.dumps(row) + "\n")
                log.flush()
                report["exposed_videos"].extend(item["window"]["video_id"] for item in used)
                print(json.dumps({k: row[k] for k in ["step", "loss", "seconds", "teacher_forcing_tokens", "padded_tokens"]}
                                 | {"windows": len(used)}), flush=True)
                del inputs, loss, prepared
                if args.checkpoint_every and pending == 0 and not last and (index + 1) % args.checkpoint_every == 0:
                    save_checkpoint(index + 1)
        if not losses:
            raise RuntimeError("No training window was used")
        report["skipped_windows"] = dict(skipped)
        optimizer.zero_grad(set_to_none=True)
        runtime.model.eval()
        adapter_dir = args.output / "adapter"
        runtime.model.save_pretrained(adapter_dir)
        saved = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(runtime.model).items()}
        base = runtime.model.unload()
        reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        loaded = get_peft_model_state_dict(reloaded)
        equal = saved.keys() == loaded.keys() and all(torch.equal(value, loaded[key].detach().cpu()) for key, value in saved.items())
        if not equal:
            raise RuntimeError("Adapter reload verification failed")
        report.update(status="learning_pilot_complete" if args.max_steps else "training_complete", actual_steps=len(losses),
                      actual_training_videos=len(set(report["exposed_videos"])),
                      optimizer_update_verified=any_update, save_reload_adapter_tensors_equal=equal,
                      first_loss=losses[0], last_loss=losses[-1], mean_loss=sum(losses) / len(losses),
                      training_seconds=sum(step_times), teacher_forcing_tokens=token_total,
                      warm_step_seconds=(sum(step_times[2:]) / len(step_times[2:]) if len(step_times) > 2 else None),
                      total_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key not in {"adapter_modules", "exposed_videos"}}), flush=True)


if __name__ == "__main__":
    main()

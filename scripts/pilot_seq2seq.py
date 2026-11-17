#!/usr/bin/env python
"""One-update T5Gemma compatibility pilot, locked to verified physical GPU0."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chaptering.inference import build_prompt

GPU_UUID = "GPU-05f1cfd1-0219-e850-913d-159b6ed72d60"
# Owned-job devices: physical GPU0 (standing authorization) and physical GPU2 (authorized
# 2026-11-17 for this project). GPU1 services and GPU3 are never used. Each device has its own lock.
AUTHORIZED_GPUS = {GPU_UUID: ("0", "chaptering-gpu0.lock"),
                   "GPU-d7e5651f-45ed-fd48-2bd4-7f38cc45e13b": ("2", "chaptering-gpu2.lock")}


@contextmanager
def gpu0_lock():
    """Exclusive lock on the one authorized physical GPU named by CUDA_VISIBLE_DEVICES (by UUID)."""
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if uuid not in AUTHORIZED_GPUS:
        raise RuntimeError("Requires the exact UUID mask of an authorized physical GPU (GPU0 or GPU2)")
    index, lock_name = AUTHORIZED_GPUS[uuid]
    lock_path = Path.home() / ".cache" / lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
        rows = [[part.strip() for part in line.split(",")] for line in output.splitlines()]
        selected = [row for row in rows if row[1] == uuid]
        if len(selected) != 1 or selected[0][0] != index or int(selected[0][2]) > 16:
            raise RuntimeError("Authorized physical GPU identity/availability check failed")
        apps = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
        if any(uuid in line for line in apps.splitlines()):
            raise RuntimeError("Authorized GPU has an existing compute workload")
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", choices=["E1", "E2"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="One canonical train record; default synthetic")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attention-implementation", choices=["eager", "sdpa", "flex_attention", "flash_attention_2"], default="eager")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"alias": args.alias, "purpose": "one-update compatibility only; not effectiveness",
              "gpu_uuid": GPU_UUID, "pid": os.getpid(), "status": "running",
              "model_directory": str(args.model_dir.resolve()), "phases_seconds": {}}
    start = time.perf_counter()
    try:
        with gpu0_lock():
            import torch
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from chaptering.evaluation import parse_outline

            torch.manual_seed(393)
            revision = (args.model_dir / "PINNED_REVISION").read_text().strip()
            report["revision"] = revision
            if args.record:
                row = json.loads(args.record.read_text().splitlines()[0])
                if row.get("split") != "train":
                    raise ValueError("Compatibility updates may use training examples only")
            else:
                row = {"video_id": "synthetic-c-pilot", "sentences": [
                    "To bake bread, mix flour with water and yeast.",
                    "Knead the dough, let it rise, then bake it in a hot oven.",
                    "Now let us discuss basic bicycle maintenance.",
                    "Check tire pressure and lubricate the chain before a long ride."],
                    "reference": {"chapter_starts": [0, 2], "chapter_titles": ["Baking bread", "Bicycle maintenance"]}}
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": build_prompt(row)}],
                                                   tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            target = json.dumps({"chapters": [{"start": s, "title": t} for s, t in
                                             zip(row["reference"]["chapter_starts"], row["reference"]["chapter_titles"])]},
                                separators=(",", ":"), ensure_ascii=False)
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if not target_ids or target_ids[-1] != tokenizer.eos_token_id:
                target_ids.append(tokenizer.eos_token_id)
            phase = time.perf_counter()
            base = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir), local_files_only=True,
                                                       dtype=torch.bfloat16, device_map={"": 0},
                                                       attn_implementation=args.attention_implementation)
            torch.cuda.synchronize()
            report["phases_seconds"]["load"] = time.perf_counter() - phase
            encoder_limit = base.config.encoder.max_position_embeddings
            decoder_limit = base.config.decoder.max_position_embeddings
            if inputs["input_ids"].shape[1] > encoder_limit or len(target_ids) > decoder_limit:
                raise ValueError("Example exceeds pinned context; truncation is forbidden")
            report.update(input_tokens=inputs["input_ids"].shape[1], target_tokens=len(target_ids),
                          encoder_limit=encoder_limit, decoder_limit=decoder_limit,
                          video_id=row["video_id"], template_source="pinned official tokenizer",
                          attention_implementation=args.attention_implementation,
                          precision="bfloat16", target_supervision="outline JSON + EOS; encoder separate")
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            labels = torch.tensor([target_ids], dtype=torch.long, device="cuda:0")

            def generate(model, label):
                phase_start = time.perf_counter()
                model.eval()
                with torch.no_grad():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
                torch.cuda.synchronize()
                text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
                if not text:
                    raise RuntimeError("Generation decoded to empty text")
                value = {"nonempty": bool(text), "generated_tokens": output.shape[-1],
                         "seconds": time.perf_counter() - phase_start,
                         "final_text": text}
                try:
                    value["chapters"] = parse_outline(text, len(row["sentences"]))
                    value["outline_status"] = "ok"
                except ValueError:
                    value["outline_status"] = "invalid_or_truncated"
                report[label] = value
                return output.detach().cpu()

            generate(base, "unadapted_generation")
            config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                                lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
            model = get_peft_model(base, config)
            adapted_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
            report["adapter_modules"] = adapted_modules
            report["adapter_coverage"] = {
                "encoder": sum("encoder" in name for name in adapted_modules),
                "decoder_self_attention": sum("decoder" in name and "self_attn" in name for name in adapted_modules),
                "decoder_cross_attention": sum("decoder" in name and "cross_attn" in name for name in adapted_modules)}
            if not all(report["adapter_coverage"].values()):
                raise RuntimeError("Expected encoder/decoder/cross-attention adapter coverage missing")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train()
            trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
            report["trainable_parameters"] = sum(value.numel() for _, value in trainable)
            optimizer = torch.optim.AdamW([value for _, value in trainable], lr=1e-4)
            phase = time.perf_counter()
            loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            if any(value.grad is not None and not torch.isfinite(value.grad).all()
                   for _, value in trainable):
                raise RuntimeError("Nonfinite adapter gradient")
            gradient_params = [(name, value) for name, value in trainable
                               if value.grad is not None and torch.count_nonzero(value.grad).item() > 0]
            if not gradient_params:
                raise RuntimeError("No nonzero adapter gradients")
            name, parameter = gradient_params[0]
            before = parameter.detach().clone()
            optimizer.step()
            if any(not torch.isfinite(value).all() for _, value in trainable):
                raise RuntimeError("Nonfinite adapter parameter after update")
            changed = torch.count_nonzero(parameter.detach() != before).item()
            if not changed:
                raise RuntimeError("Adapter parameter did not update")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            report.update(loss=loss.item(), nonzero_gradient_tensors=len(gradient_params),
                          updated_parameter=name, changed_parameter_elements=changed)
            report["phases_seconds"]["one_update"] = time.perf_counter() - phase
            model.config.use_cache = True
            adapted_output = generate(model, "adapted_generation")
            adapter_dir = args.output / "adapter"
            model.save_pretrained(adapter_dir)
            saved_state = {key: value.detach().cpu().clone()
                           for key, value in get_peft_model_state_dict(model).items()}
            base = model.unload()
            reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            loaded_state = get_peft_model_state_dict(reloaded)
            report["save_reload_adapter_tensors_equal"] = (
                saved_state.keys() == loaded_state.keys()
                and all(torch.equal(value, loaded_state[key].detach().cpu())
                        for key, value in saved_state.items()))
            if not report["save_reload_adapter_tensors_equal"]:
                raise RuntimeError("Adapter tensors changed after save/reload")
            reloaded_output = generate(reloaded, "reloaded_generation")
            report["save_reload_generation_equal"] = torch.equal(adapted_output, reloaded_output)
            if not report["save_reload_generation_equal"]:
                raise RuntimeError("Deterministic generation changed after save/reload")
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["status"] = "compatibility_pass"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
    report["total_seconds"] = time.perf_counter() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "adapter_modules"}), flush=True)
    return 0 if report["status"] == "compatibility_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

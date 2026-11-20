"""Run the bundled model benchmarks from the project root."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SUFFIX = ".exe" if os.name == "nt" else ""
BASELINE = f"baseline/picolm_baseline{SUFFIX}"
OPT = f"./picolm_opt{SUFFIX}"
SCALAR = f"./picolm_scalar{SUFFIX}"
TINYLLAMA = "datasets/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
SCHEMAS = "datasets/json_eval_prompts.jsonl"
JSON_ARGS = ["-n", "32", "-t", "0", "-j", "4", "--json"]
JSON_CACHE = Path("build/json_baseline.json")
OPT_JSON_CACHE = Path("build/json_optimized.json")
VERIFICATION_LOG = Path("reports/verification.log")
VERIFICATION_MANIFEST = Path("reports/verification_manifest.json")


def run(binary, model, args, *, prompt=None, timeout=180):
    # Pipe prompts so embedded double quotes and backslashes survive on Windows.
    started = time.perf_counter()
    result = subprocess.run(
        [binary, model, *args],
        input=prompt.encode("utf-8") if prompt is not None else b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return result, (time.perf_counter() - started) * 1000


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def non_json_number(value):
    raise ValueError(value)


def read_json_strict(output):
    return json.loads(output.decode("utf-8"), parse_constant=non_json_number)


def json_cases():
    return [json.loads(line) for line in Path(SCHEMAS).read_text(encoding="utf-8").splitlines()]


def bench_record(result):
    for line in reversed(result.stderr.splitlines()):
        if line.startswith((b'{"prefill_tokens"', b'{"finish_reason"')):
            return json.loads(line)
    return {}


def bench_json(binary):
    rows = []
    for case in json_cases():
        args = ["--bench", *JSON_ARGS] if binary == OPT else JSON_ARGS
        result, _ = run(binary, TINYLLAMA, args, prompt=case["prompt"])
        try:
            value = read_json_strict(result.stdout)
            valid = result.returncode == 0
        except (UnicodeError, ValueError):
            value = None
            valid = False
        metrics = bench_record(result) if binary == OPT else {}
        suffix = metrics.get("json_suffix_bytes")
        finish = metrics.get("finish_reason", "process_error" if result.returncode else None)
        if binary == OPT and result.returncode == 0 and finish not in {"budget", "eos", "grammar_complete"}:
            raise ValueError("Missing or invalid optimized JSON finish reason")
        if suffix and finish != "budget":
            raise ValueError("JSON closure was reported outside budget exhaustion")
        rows.append({
            "id": case["id"],
            "exit_code": result.returncode,
            "valid": valid,
            "object": valid and isinstance(value, dict),
            "suffix_bytes": suffix,
            "finish_reason": finish,
        })
        print(f"{binary} {case['id']} valid={valid}", flush=True)
    return rows


def json_provenance(binary):
    return {
        "binary_sha256": file_sha256(binary),
        "model_sha256": file_sha256(TINYLLAMA),
        "prompts_sha256": file_sha256(SCHEMAS),
        "args": JSON_ARGS,
    }


PREFILL = re.compile(rb"Prefill: (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
DECODE = re.compile(rb"Generation: (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
STORIES = ["datasets/stories15M-q4_0.gguf", "datasets/stories15M-q8_0.gguf"]
WIKITEXT = "datasets/wikitext2_eval.txt"


def successful(result, label):
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed ({result.returncode}): {result.stderr.decode('utf-8', 'replace')}")


def generated_stats(result):
    metrics = bench_record(result)
    if "prefill_tokens" not in metrics:
        raise ValueError("Missing --bench diagnostics")
    return metrics


def parse_baseline(result):
    prefill = PREFILL.search(result.stderr)
    decode = DECODE.search(result.stderr)
    if prefill is None or decode is None:
        raise ValueError("Missing upstream timings")
    return {
        "prompt_tokens": int(prefill[1]),
        "prefill_ms": float(prefill[2]) * 1000,
        "prefill_tok_s": float(prefill[3]),
        "generated_tokens": int(decode[1]),
        "decode_ms": float(decode[2]) * 1000,
        "decode_tok_s": float(decode[3]),
    }


def normalized_output(result):
    return result.stdout.replace(b"\r\n", b"\n")


def comparison_run(binary, model, prompt, args):
    options = args if binary == BASELINE else ["--bench", *args]
    result, elapsed = run(binary, model, options, prompt=prompt)
    successful(result, binary)
    metrics = parse_baseline(result) if binary == BASELINE else generated_stats(result)
    return {
        "metrics": metrics,
        "process_elapsed_ms": round(elapsed, 3),
        "output_sha256": hashlib.sha256(normalized_output(result)).hexdigest(),
    }


def median_metric(rows, name):
    return statistics.median(row["metrics"][name] for row in rows)


def prompt_with_tokens(count):
    # TinyLlama tokenizes each space-prefixed 'the' as one piece, plus BOS.
    return " ".join(["the"] * (count - 1))


def bench_tinyllama(repeats):
    results = []
    for n in (32, 128):
        for threads in (1, 4):
            opts = ["-n", "16", "-t", "0", "-j", str(threads)]
            prompt = prompt_with_tokens(n)
            row = {"prompt_tokens": n, "threads": threads, "baseline": [], "optimized": []}
            for trial in range(repeats):
                order = [(BASELINE, "baseline"), (OPT, "optimized")]
                if trial % 2:
                    order.reverse()
                for binary, label in order:
                    sample = comparison_run(binary, TINYLLAMA, prompt, opts)
                    if sample["metrics"]["prompt_tokens"] != n:
                        raise RuntimeError(f"Expected {n} prompt tokens for {binary}: {sample}")
                    row[label].append(sample)
                    print(f"TinyLlama {label}, prompt={n}, threads={threads}, run={trial + 1}", flush=True)
            row["text_equivalent"] = (
                len({x["output_sha256"] for x in row["baseline"] + row["optimized"]}) == 1
            )
            results.append(row)
    return results


def compile_trace(cc):
    build = Path("build")
    build.mkdir(exist_ok=True)
    # Wrap just the sampler call in a separate build of the pinned upstream sources.
    shim = build / "trace_baseline.c"
    shim.write_text(
        '#include "../baseline/sampler.h"\n'
        '#include <stdio.h>\n'
        'static int trace_sample(sampler_t *s, float *logits, int vocab);\n'
        '#define sampler_sample trace_sample\n'
        '#include "../baseline/picolm.c"\n'
        '#undef sampler_sample\n'
        'static int trace_sample(sampler_t *s, float *logits, int vocab) {\n'
        '    int token = sampler_sample(s, logits, vocab);\n'
        '    fprintf(stderr, "TRACE_TOKEN %d\\n", token);\n'
        '    return token;\n'
        '}\n',
        encoding="ascii",
    )
    trace = build / f"picolm_baseline_trace{SUFFIX}"
    inputs = ["model", "tensor", "quant", "tokenizer", "sampler", "grammar"]
    command = [cc, "-O3", "-std=c11", "-D_GNU_SOURCE", "-march=native", "-o", trace.as_posix(),
               shim.as_posix(), *(f"baseline/{name}.c" for name in inputs), "-lm"]
    command.append("-static" if os.name == "nt" else "-pthread")
    result = subprocess.run(command, capture_output=True, timeout=120)
    (build / "trace_baseline_build.log").write_bytes(result.stdout + result.stderr)
    successful(result, "baseline sampler trace build")
    return trace.as_posix()


def trace_baseline(tinyllama, cc):
    binary = compile_trace(cc)
    # Trace each distinct prompt with four threads and check against both thread counts.
    for length in (32, 128):
        args = ["-n", "16", "-t", "0", "-j", "4"]
        process, _ = run(binary, TINYLLAMA, args, prompt=prompt_with_tokens(length))
        successful(process, "upstream trace")
        ids = [int(i) for i in re.findall(rb"^TRACE_TOKEN (\d+)\r?$", process.stderr, re.MULTILINE)]
        digest = hashlib.sha256(normalized_output(process)).hexdigest()
        if not ids or len(ids) != parse_baseline(process)["generated_tokens"]:
            raise RuntimeError(f"Upstream trace is incomplete for {length} tokens")
        for row in tinyllama:
            if row["prompt_tokens"] != length:
                continue
            row["upstream_token_ids"] = ids
            row["trace_text_matches_baseline"] = all(digest == x["output_sha256"] for x in row["baseline"])
            row["token_ids_equivalent"] = all(ids == x["metrics"]["token_ids"] for x in row["optimized"])
        print(f"Upstream token trace for prompt={length}: {len(ids)} IDs", flush=True)


def bench_kv(tinyllama, repeats):
    opts = ["-n", "16", "-t", "0", "-j", "4"]
    fp16 = next(row["optimized"] for row in tinyllama
                if row["prompt_tokens"] == 128 and row["threads"] == 4)
    prompt = prompt_with_tokens(128)
    int8 = [comparison_run(OPT, TINYLLAMA, prompt, [*opts, "--kv-quant", "int8"]) for _ in range(repeats)]
    print("TinyLlama KV format comparison complete", flush=True)
    return {"fp16": fp16, "int8": int8}


def bench_threadpool(tinyllama, repeats):
    opts = ["-n", "16", "-t", "0", "-j", "4", "--no-threadpool"]
    persistent = next(row["optimized"] for row in tinyllama
                      if row["prompt_tokens"] == 32 and row["threads"] == 4)
    transient = [comparison_run(OPT, TINYLLAMA, prompt_with_tokens(32), opts) for _ in range(repeats)]
    print("TinyLlama thread-pool comparison complete", flush=True)
    return {"persistent": persistent, "transient": transient}


def bench_stories(repeats):
    rows = []
    for model in STORIES:
        opts = ["-n", "64", "-t", "0", "-j", "4"]
        row = {"model": Path(model).name, "simd": [], "scalar": []}
        for trial in range(repeats):
            order = [(OPT, "simd"), (SCALAR, "scalar")]
            if trial % 2:
                order.reverse()
            for binary, label in order:
                row[label].append(comparison_run(binary, model, "Once upon a time", opts))
        ids = {tuple(sample["metrics"]["token_ids"]) for sample in row["simd"] + row["scalar"]}
        row["token_ids_equivalent"] = len(ids) == 1
        print(f"Stories SIMD/scalar complete: {Path(model).name}", flush=True)
        rows.append(row)
    return rows


def bench_ppl(byte_budget, timeout):
    text = Path(WIKITEXT).read_bytes().replace(b"\r\n", b"\n")
    prefix = text[:byte_budget]
    if len(text) > byte_budget:
        end = max((prefix.rfind(whitespace) for whitespace in (b" ", b"\n", b"\t")), default=-1)
        if end >= 0:
            prefix = prefix[:end + 1]
    prefix.decode("utf-8")
    build = Path("build")
    build.mkdir(exist_ok=True)
    metrics = {"input_bytes": len(prefix), "input_sha256": hashlib.sha256(prefix).hexdigest()}
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix="ppl-", suffix=".txt", dir=build, delete=False) as f:
            path = Path(f.name)
            f.write(prefix)
        for quant in ("fp16", "int8"):
            result, _ = run(
                OPT, TINYLLAMA, ["-j", "4", "--ppl", path.resolve().relative_to(ROOT).as_posix(), "--kv-quant", quant],
                timeout=timeout,
            )
            successful(result, f"perplexity {quant}")
            metrics[quant] = read_json_strict(result.stdout)
            print(f"WikiText-2 {quant} complete", flush=True)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    if metrics["fp16"]["scored_tokens"] != metrics["int8"]["scored_tokens"]:
        raise RuntimeError("Perplexity modes scored different token counts")
    metrics["ppl_delta"] = metrics["int8"]["perplexity"] - metrics["fp16"]["perplexity"]
    return metrics


def json_rows(binary, cache, reuse):
    provenance = json_provenance(binary)
    if reuse and cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        info = cached["provenance"]
        if "baseline_sha256" in info:
            info["binary_sha256"] = info.pop("baseline_sha256")
        if info == provenance and [r["id"] for r in cached["rows"]] == [c["id"] for c in json_cases()]:
            if binary == BASELINE:
                for row in cached["rows"]:
                    row.setdefault("finish_reason", "process_error" if row["exit_code"] else None)
                return cached["rows"]
            if all("finish_reason" in row for row in cached["rows"]):
                return cached["rows"]
    rows = bench_json(binary)
    if json_provenance(binary) != provenance:
        raise RuntimeError(f"JSON benchmark inputs changed for {binary}")
    write_json(cache, {"provenance": provenance, "rows": rows})
    return rows


def benchmark_inputs():
    inputs = [BASELINE, OPT, SCALAR, TINYLLAMA, SCHEMAS, WIKITEXT, *STORIES]
    for pattern in ("*.c", "*.h"):
        inputs.extend(sorted(Path("baseline").glob(pattern)))
    return {Path(path).as_posix(): file_sha256(path) for path in inputs}


def verified_log(binaries):
    try:
        manifest = json.loads(VERIFICATION_MANIFEST.read_text(encoding="utf-8"))
        if (manifest.get("binaries") == binaries and
                manifest.get("log_sha256") == file_sha256(VERIFICATION_LOG)):
            return {"file": VERIFICATION_LOG.name, "sha256": manifest["log_sha256"]}
    except (OSError, ValueError, AttributeError):
        pass
    return None


def full_suite(args):
    inputs = benchmark_inputs()
    binaries = {name: inputs[Path(path).as_posix()]
                for name, path in (("baseline", BASELINE), ("optimized", OPT), ("scalar", SCALAR))}
    tinyllama = bench_tinyllama(args.repeats)
    kv = bench_kv(tinyllama, args.repeats)
    pool = bench_threadpool(tinyllama, args.repeats)
    stories = bench_stories(max(args.repeats, 3))
    ppl = bench_ppl(args.ppl_bytes, args.ppl_timeout)
    trace_baseline(tinyllama, args.cc)
    baseline_json = json_rows(BASELINE, JSON_CACHE, args.reuse_json)
    opt_json = json_rows(OPT, OPT_JSON_CACHE, args.reuse_json)
    after = benchmark_inputs()
    if inputs != after:
        changed = sorted(path for path in inputs.keys() | after.keys() if inputs.get(path) != after.get(path))
        raise RuntimeError(f"Benchmark inputs changed during the run: {', '.join(changed)}")
    report = {
        "cpu": host_cpu(),
        "system": platform.system(),
        "python": platform.python_version(),
        "repeats_tinyllama": args.repeats,
        "repeats_stories": max(args.repeats, 3),
        "baseline_sha256": binaries["baseline"],
        "optimized_sha256": binaries["optimized"],
        "scalar_sha256": binaries["scalar"],
        "inputs_sha256": inputs,
        "verification": verified_log(binaries),
        "tinyllama": tinyllama,
        "kv": kv,
        "threadpool": pool,
        "stories": stories,
        "perplexity": ppl,
        "json": {"baseline": baseline_json, "optimized": opt_json, "options": JSON_ARGS},
    }
    write_json("reports/benchmark_summary.json", report)
    render_report(report)


def host_cpu():
    if os.name == "nt":
        import winreg
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as cpu:
            return winreg.QueryValueEx(cpu, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()



def render_report(report):
    lines = [
        "# PicoLM Edge 基准报告", "",
        f"环境为 {report['system']}，处理器为 {report['cpu']}，Python {report['python']}。",
        f"TinyLlama 每组运行 {report['repeats_tinyllama']} 次，Stories15M 每组运行 "
        f"{report['repeats_stories']} 次。表中取中位数。", "",
        "## TinyLlama-1.1B Q4_K_M", "",
        "上下文为模型默认值 2048。原始基线用于计时，单独编译的上游采样打点版本用于读取 "
        "token ID。TTFT 从预填充开始计算。基线取日志中精确到 0.01 秒的预填充时长，"
        "优化版计到首个 token 选出。速率单位为 token/s，TTFT 为毫秒。", "",
        "| 提示 token | 线程 | 基线预填充 | 优化预填充 | 加速 | 基线生成 | 优化生成 | 加速 | "
        "基线 TTFT | 优化 TTFT | 生成 token | 文本与 token ID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["tinyllama"]:
        b, o = row["baseline"], row["optimized"]
        bp, op = median_metric(b, "prefill_tok_s"), median_metric(o, "prefill_tok_s")
        bd, od = median_metric(b, "decode_tok_s"), median_metric(o, "decode_tok_s")
        n = median_metric(o, "generated_tokens")
        same = row["text_equivalent"] and row["trace_text_matches_baseline"] and row["token_ids_equivalent"]
        lines.append(
            f"| {row['prompt_tokens']} | {row['threads']} | {bp:.2f} | {op:.2f} | {op / bp:.2f}× | "
            f"{bd:.2f} | {od:.2f} | {od / bd:.2f}× | {median_metric(b, 'prefill_ms'):.0f} | "
            f"{median_metric(o, 'ttft_ms'):.1f} | {n:g} | {'一致' if same else '有差异'} |"
        )

    kv = report["kv"]
    saved = 100 * (1 - median_metric(kv["int8"], "kv_cache_bytes") /
                   median_metric(kv["fp16"], "kv_cache_bytes"))
    lines.extend([
        "", "## KV 缓存与线程池", "",
        "KV 对照使用 128 token 提示、4 线程及 2048 上下文。KV 分配量按缓存向量和尺度计算。"
        "峰值 RSS 来自进程工作集，会随权重驻留和上下文实际触及的页变化。", "",
        "| KV 类型 | KV 分配（MiB） | 峰值 RSS（MiB） | 预填充（token/s） | 生成（token/s） |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, rows in kv.items():
        lines.append(
            f"| {name.upper()} | {median_metric(rows, 'kv_cache_bytes') / 1048576:.3f} | "
            f"{median_metric(rows, 'peak_rss_mb'):.3f} | {median_metric(rows, 'prefill_tok_s'):.2f} | "
            f"{median_metric(rows, 'decode_tok_s'):.2f} |"
        )
    lines.extend([
        "", f"INT8 的 KV 分配量下降 {saved:.3f}%。", "",
        "线程池对照使用 32 token 提示与 4 线程，两组使用同一个优化版可执行文件。", "",
        "| 模式 | 预填充（token/s） | 生成（token/s） | TTFT（ms） |",
        "|---|---:|---:|---:|",
    ])
    for name, rows in report["threadpool"].items():
        label = {"persistent": "持久线程池", "transient": "临时线程"}[name]
        lines.append(
            f"| {label} | {median_metric(rows, 'prefill_tok_s'):.2f} | "
            f"{median_metric(rows, 'decode_tok_s'):.2f} | {median_metric(rows, 'ttft_ms'):.1f} |"
        )

    ppl = report["perplexity"]
    lines.extend([
        "", "## WikiText-2 困惑度", "",
        f"评估文件开头截至完整单词的 {ppl['input_bytes']} 个 UTF-8 字节，"
        f"每种 KV 配置评分 {ppl['fp16']['scored_tokens']} 个预测 token。",
        "", "| KV 类型 | 平均 NLL | 困惑度 | 计算时间（秒） |",
        "|---|---:|---:|---:|",
    ])
    for name in ("fp16", "int8"):
        row = ppl[name]
        lines.append(f"| {name.upper()} | {row['nll']:.8f} | {row['perplexity']:.8f} | "
                     f"{row['elapsed_ms'] / 1000:.2f} |")
    lines.extend(["", f"INT8 的困惑度差值为 {ppl['ppl_delta']:+.8f}。"])

    lines.extend([
        "", "## JSONSchemaBench 提示", "",
        "使用数据文件的原始 `prompt` 字段、固定贪心配置和 `-n 32`。"
        "使用 Python `json.loads()` 和 UTF-8 严格解码判定 JSON 语法。"
        "只有退出码为 0 且语法完整的对象才计入对象数。补全计数仅统计耗尽 token 预算后的闭合。"
        "本实验的指标限定为语法有效率，Schema 满足率未测。", "",
        "| 版本 | 有效 JSON | 对象 | 进程失败 | 使用闭合后缀 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, label in (("baseline", "原始基线"), ("optimized", "优化版")):
        rows = report["json"][key]
        valid = sum(row["valid"] for row in rows)
        objects = sum(row["object"] for row in rows)
        failures = sum(row["exit_code"] != 0 for row in rows)
        suffix = sum(row.get("finish_reason") == "budget" and (row.get("suffix_bytes") or 0) > 0
                     for row in rows)
        lines.append(f"| {label} | {valid}/{len(rows)} | {objects}/{len(rows)} | "
                     f"{failures} | {suffix if key == 'optimized' else '—'} |")

    lines.extend([
        "", "## Stories15M Q4_0 与 Q8_0", "",
        "原始基线的 MHA 加载问题依据任务随附的观察与源代码。优化版和标量版均使用 4 线程、"
        "FP16 KV 缓存、相同 prompt 和 `-n 64`。", "",
        "| 模型 | 标量预填充 | SIMD 预填充 | 标量生成 | SIMD 生成 | 生成加速 | token ID |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in report["stories"]:
        scalar, simd = row["scalar"], row["simd"]
        sd, nd = median_metric(scalar, "decode_tok_s"), median_metric(simd, "decode_tok_s")
        same = "一致" if row["token_ids_equivalent"] else "有差异"
        lines.append(
            f"| {row['model']} | {median_metric(scalar, 'prefill_tok_s'):.2f} | "
            f"{median_metric(simd, 'prefill_tok_s'):.2f} | {sd:.2f} | {nd:.2f} | "
            f"{nd / sd:.2f}× | {same} |"
        )
    lines.extend([
        "", "## 复现", "",
        "Windows 运行以下命令。先让 MinGW-w64 GCC 和 Python 在当前会话的 PATH 可用。", "",
        "```powershell",
        "mingw32-make -j4 native scalar",
        "mingw32-make -j4 test",
        "python scripts/run_benchmarks.py",
        "```", "",
        "逐例指标、二进制 SHA-256、生成 token ID 和输出摘要保存在 `benchmark_summary.json`。",
    ])
    if report.get("verification"):
        lines.extend(["", "与参测二进制哈希匹配的验收证据见 [构建与测试日志](verification.log)。"])
    write_text("reports/benchmark_report.md", "\n".join(lines) + "\n")


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--json-baseline", action="store_true", help="run the baseline JSON pass")
    stage.add_argument("--json-optimized", action="store_true", help="run the optimized JSON pass")
    parser.add_argument("--reuse-json", action="store_true", help="reuse JSON syntax runs with matching hashes")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ppl-bytes", type=int, default=768)
    parser.add_argument("--ppl-timeout", type=float, default=180, help="timeout per perplexity run in seconds")
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    args = parser.parse_args()
    if args.repeats < 1 or args.ppl_bytes < 64:
        parser.error("--repeats must be positive and --ppl-bytes must be at least 64")
    if not math.isfinite(args.ppl_timeout) or args.ppl_timeout <= 0:
        parser.error("--ppl-timeout must be positive and finite")
    if args.json_baseline:
        json_rows(BASELINE, JSON_CACHE, False)
    elif args.json_optimized:
        json_rows(OPT, OPT_JSON_CACHE, False)
    else:
        full_suite(args)


if __name__ == "__main__":
    main()

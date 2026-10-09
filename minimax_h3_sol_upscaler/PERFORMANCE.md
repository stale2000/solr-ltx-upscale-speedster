# H3 SoL Upscaler performance findings

**Two RTX 5090s upscaled a 15.08-second H3 clip to 1184×666 in 13.61 seconds end-to-end: 1.11× real time and 17% faster than one GPU.** A separate live shared-VAE run delivered 42 consecutive 15-second clips. These are warm benchmark results and a finite live smoke test, not a long-duration service guarantee.

**What it does:** The upscaler takes low-resolution MiniMax-H3 video as input and uses SoL Refiner to produce higher-resolution output—for example, 512×288 to 1184×666—while aiming to preserve the scene and motion and carry the audio through unchanged. The goal is continuous upscaling fast enough to keep pace with incoming clips.

## Scope

The packaging changes do not affect inference speed. The results below summarize measured benchmark runs; raw receipts and media are not included here. Results vary with workload, prompt, GPU, and runtime and may not reproduce on other systems.

“Original” below means the slower reference path or control named for each comparison, not a universal baseline. Do not combine results from different resolutions, GPU models, prompts, frame counts, or precision modes into one total savings claim.

## Measured gains

| Optimization | Reported comparison | Measured result | Quality and scope |
|---|---|---|---|
| Distilled `pruna_fast` decoder | 1024×576 warm MP4-ready: 24.64 s reference diffusion path vs 7.43 s fast path | 17.21 s less (69.8%); about 3.3× faster | Changes fine detail and has a small temporal sharpness variation. This is the largest reported warm-path gain, but it is not quality-equivalent. |
| Dynamic NVFP4 transformer weights and fused activation quantizer | 1184×666 warm: 12.997 s MXFP8 vs 12.371 s NVFP4 | 0.626 s less (4.8%); peak allocated VRAM 33.43 → 25.16 GiB (8.27 GiB less), reserved VRAM 5.56 GiB less | Blackwell only. Full-video comparison against MXFP8: 26.17 dB PSNR / 0.8895 SSIM, with visible generated-detail differences. The default favors speed and memory, not output parity. |
| Repeated transformer-block compilation | 1280×720 warm worker median: 15.48 s eager vs 14.41 s compiled | 1.07 s less (6.9%) | Compiled mode is the `pruna_fast` default. First compile is a cold-start cost; measured gains are warm. |
| Repeated transformer-block compilation | 1184×666 warm worker median: 14.56 s eager vs 13.62 s compiled | 0.94 s less (6.5%) based on the displayed medians | Earlier wording listed 0.84 s / 5.8%, which does not match those medians; this summary corrects the arithmetic. MP4s differed: 40.69 dB PSNR / 0.9884 SSIM. |
| CUDA YUV420P packing for NVENC | 1280×720 aggregate export: 1.862 s RGB24 vs 1.180 s YUV420P; total worker: 18.092 s vs 17.376 s | Export 0.682 s less (36.6%); total 0.716 s less (4.0%). Raw pipe and four-frame staging bytes halved, 11.06 → 5.53 MB per batch. | Output was 48.31 dB PSNR / 0.9932 SSIM against RGB24; reviewed contact sheets showed no visible difference. All 647 source audio packet payloads matched. Not pixel-identical. |
| Exact fixed-prompt conditioning artifact | 1184×666 peak allocated VRAM: 78.7 → 50.5 GiB | 28.2 GiB less (35.8%); warm time changed only 0.063 s (0.42%), within drift | Same-prompt MP4 was byte-identical. Benefit is memory headroom and avoiding repeated text encoding, not a demonstrated throughput gain. Different prompts require different artifacts. |
| MXFP8 transformer weights (manual alternative) | 1184×666 vs compiled BF16 | 0.91 s less (6.65%); peak allocated VRAM 50.52 → 33.43 GiB (17.09 GiB less) | Encoded output comparison: 32.46 dB PSNR / 0.9663 SSIM. MXFP8 is not the current default. |
| Two-GPU split on 2×RTX 5090 | 15.083-second clip: single-GPU warm end-to-end flow median 16.43 s vs split median 13.61 s | **1.11× real-time throughput** (15.083 ÷ 13.61); 2.82 s (17.1%) less per clip than the single-GPU arm. About 4.4 clips/minute or 66.5 media-seconds/minute. | Real ComfyUI workflow; only two warm samples per profile. Each GPU inferred one chunk in 6.50–6.98 s; chunk assembly took about 3.8 s. Outputs had 362 frames and preserved audio. The join looked continuous in one limited contact-sheet review. |
| Continuous 2×RTX 5090 shared-VAE live run | 42 admitted 15-second clips; 630 one-second HLS segments | Upscale stage times 14.40–31.80 s, 17.25 s median; the player had about 101 s buffered and was not stalled at the final check | This confirms the live path ran continuously during the observation window. It was a smoke test, not a one-hour or low-latency qualification; the median stage time exceeded the 15.083 s clip duration. The 2×5090 upscaler used one GPU for VAE/decoder work and one for transformer work. |
| Earlier shared-VAE three-GPU comparison | 15-second clip: one RTX 5090 at 17.46 s, two-GPU MP4 split at 18.68 s, shared-VAE three-GPU at 11.07 s | Shared-VAE was 6.39 s (36.6%) faster than the single-GPU arm in that run | One scene with a prompt/source semantic mismatch. This is a different host/run from the 2×5090 results above; results are not contradictory or directly interchangeable. |

The fixed-conditioning and precision memory figures come from different benchmark arms and must not be added together. Likewise, the decoder, compile, NVFP4, and YUV420P percentages are separate comparisons, not cumulative percentages.

Use the two-GPU split row for the measured per-clip throughput estimate. The 17.25-second median in the separate 42-clip live run is an upscale-stage timing for a different shared-VAE profile and 1036×592 output; it is not the two-GPU split flow time.

## Other implemented optimizations

- **Persistent isolated worker:** keeps model weights resident on CUDA between node calls and keeps the host's PyTorch 2.14 separate from the worker's pinned PyTorch 2.13/NATTEN stack. This avoids repeated model-load costs on warm requests; no single controlled load-time saving is claimed here.
- **Prompt reuse:** a bounded exact-prompt LRU stores up to four embedding entries / 64 MiB; the tested prompt used 8.4 MB. Hits return private tensor copies. The measured warm worker gain was only 0.30%, below 1.49% control drift, and outputs were byte-identical. The separate fixed-conditioning artifact is useful when the prompt never changes and has a measured memory benefit above.
- **Pruna model loading and output buffers:** removes an unused 637,873,794-byte (0.594 GiB) Pruna encoder before moving the VAE to CUDA. The output path also removes a temporary frame copy. The paired MP4s were byte-identical; encoder removal reduced measured peak allocated/reserved memory by 0.594 GiB, while output-copy removal did not lower the run's peak high-water mark. No throughput gain was established.
- **Bounded tiled decoding and buffer lifetime:** uses overlap tiles for the reference diffusion decoder, selects a full spatial tile for smaller output sizes, releases completed conditioning buffers before decode, and avoids retaining a duplicate LTX VAE decoder in `pruna_fast`. These reduce redundant work or residency, but the available benchmark records do not isolate a savings number for each.
- **Fused rotary math and native NATTEN:** the reference diffusion decoder uses cached rotary tables, a fused FP32 rotary kernel configured to preserve eager rounding, and a pinned native attention backend. These are compatibility/performance paths; no standalone A/B savings figure is claimed here.
- **Shared-VAE transfers:** the experimental mode can enqueue latent copies on a peer CUDA stream, signal consumers with events, and retain source buffers until each copy completes. It separates copy duration from host enqueue time; neither duration should be mistaken for end-to-end savings. The node's existing default remains blocking.
- **Media correctness:** output packing follows Diffusers' round-to-even conversion. Video export is bounded by frame count rather than a global duration limit, which preserves audio packets with negative priming timestamps. The audio fix recovered all 647 source packet payloads in the tested samples; it is a correctness fix, not a speed claim.
- **Chunk consistency:** shared-VAE mode slices one clip-wide seeded latent-noise field across chunks, so matching overlap positions use the same noise. This improves seam consistency; it does not guarantee pixel parity with whole-video inference.

## Experiments without a reliable speed win

The benchmarks did not establish a throughput improvement from prompt-cache hits, overlapping CPU input preprocessing with prompt encoding, `torch.inference_mode()` around Pruna, video-only transformer forwarding, asynchronous NVENC staging, removing the Pruna output copy, using `torch.aminmax` for output validation, or changing NVENC to p1/p2. These options remain disabled or are used only where they serve memory, correctness, or optional experimentation. The Pruna decoder compilation trial was slower and failed its SSIM gate.

An experimental 1024-pixel diffusion-decoder tile reached 20.85 s MP4-ready versus about 24.88 s for the instrumented 768-pixel tiled run, but used 89.94 GiB allocated and 93.21 GiB reserved out of 95.59 GiB visible to PyTorch. It leaves little headroom and is not the default.

## Operating caveats

- Cold downloads, model load, and `torch.compile` can dominate the first request. Warm medians are the relevant figures for a persistent service.
- Runs used hosts with different GPU clocks or power limits. A reported 600 W hardware maximum is not proof of 600 W enforced under load. Treat absolute latency comparisons across hosts cautiously.
- A separate 1184×666 live OBS run had 54 of 57 warm worker calls finish within a 15.083-second input duration, but its live buffer still reported stalls. The later 2×5090 shared-VAE smoke run admitted 42 clips and ended with a healthy buffered player; neither observation qualifies an hour-long or low-latency service.
- The fast decoder, NVFP4, quantization, tiling, and compilation can change output pixels. Check temporal consistency and visual quality on representative clips before choosing a mode.

## Benchmark method for future comparisons

Compare the same input, prompt, seed, frame count, output dimensions, audio, GPU, and runtime pins. Separate cold load/compile from warm execution. Record end-to-end time until a valid MP4 is ready, multiple warm samples and spread, peak allocated/reserved VRAM, output/audio checks, and visual/temporal quality. The benchmark target was under 15 seconds MP4-ready for a 15-second, 360-frame 1024×576 output; that target is workload-specific, and a decoder-only timer does not meet it.

## Overall measured impact

- **Fastest reported warm single-GPU path:** the distilled `pruna_fast` decoder reduced the 1024×576 reference-path time from 24.64 to 7.43 seconds, saving 17.21 seconds (69.8%, about 3.3×). This changes fine detail and is not quality-equivalent to the reference diffusion decoder.
- **Two-5090 split throughput:** one 15.083-second 1184×666 clip completed in a 13.61-second warm end-to-end flow median, or **1.11× real time**. Against the same benchmark's single-GPU arm, it saved 2.82 seconds (17.1%). The continuous shared-VAE run separately processed 42 clips during its observation window; it was not a throughput qualification.
- **Largest isolated VRAM reduction:** exact fixed-prompt conditioning reduced peak allocated VRAM by 28.2 GiB in its measured arm, but did not produce a meaningful warm speed gain. In the current default precision comparison, NVFP4 saved 8.27 GiB allocated and 5.56 GiB reserved versus MXFP8. The Pruna encoder removal saved another 0.594 GiB in its own comparison.
- **Export improvement:** CUDA YUV420P packing reduced aggregate export by 0.682 seconds and total worker time by 0.716 seconds (4.0%) versus RGB24 in its 1280×720 A/B.

There is no valid grand total that adds these figures: they use different resolutions, GPUs, precision/decoder modes, and control arms, and some optimizations overlap. The headline is therefore **up to 69.8% lower warm time for the fast-decoder comparison, 1.11× real-time in the measured two-GPU split profile, and up to 28.2 GiB lower allocated VRAM in a separate fixed-prompt memory comparison**—not one cumulative speed or memory number.

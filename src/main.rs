mod fast_attention;
mod fast_norm;
mod geometry;
mod host;
mod input;
mod network;
mod ops;
mod output;
mod progress;
mod schema;
use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, DeviceKind, FloatDType};
use clap::{Parser, Subcommand, ValueEnum};
use host::{Packed, read, upload};
use input::{Input, Rng};
use output::Map;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Standard,
    Pbr,
}
#[derive(Parser)]
#[command(about="LINO-UniPS normal and PBR inference with Burn",version=env!("LINO_REVISION"))]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(long,global=true,default_value="auto",value_parser=["auto","vulkan","dx12","metal","cuda"])]
    api: String,
    #[arg(long, global = true)]
    cpu: bool,
    #[arg(long, global = true, conflicts_with = "cpu")]
    integrated: bool,
    #[arg(long,global=true,default_value="auto",value_parser=["auto","bf16","fp16","fp32"])]
    precision: String,
    /// Independent per-image encoder batch; global attention always sees every light.
    #[arg(long, global = true, default_value_t = 2)]
    encoder_batch: usize,
    /// Semantic pixel sampling group size. Changing this may change predictions.
    #[arg(long, global = true, default_value_t = 2048)]
    pixel_batch: usize,
    /// GEMM fallback score-temporary budget; fused attention does not allocate dense scores.
    #[arg(long, global = true, default_value_t = 64)]
    score_budget_mib: usize,
    /// Estimated queued temporary-work budget; live model/features are separate.
    #[arg(long, global = true, default_value_t = 64)]
    inflight_mib: usize,
    /// Use the tensor-operation fallbacks for attention, geometry and LayerNorm.
    #[arg(long, global = true)]
    reference_kernels: bool,
    /// Print block timings and write profile.json (wall-clock timings, not device timestamps).
    #[arg(long, short = 'v', global = true)]
    verbose: bool,
    /// Disable terminal progress and periodic progress lines in redirected output.
    #[arg(long, global = true)]
    no_progress: bool,
}
#[derive(Subcommand)]
enum Command {
    Infer {
        /// .safetensors, or .pth converted with the bundled Python checkpoint exporter.
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "standard")]
        mode: Mode,
        #[arg(long,visible_aliases=["nb_img","nb-img","num-images"],default_value_t=-1,allow_hyphen_values=true)]
        images: isize,
        #[arg(long)]
        batch: bool,
        #[arg(long, default_value = "L")]
        prefix: String,
        #[arg(long, default_value_t = 42)]
        seed: u32,
        /// eval matches the randomized mean/max normalization; max matches the demo path.
        #[arg(long,default_value="eval",value_parser=["eval","max"])]
        normalization: String,
        #[arg(long, default_value_t = 5632)]
        max_resolution: usize,
        #[arg(long,value_delimiter=',',default_value="exr,png",value_parser=["exr","png"])]
        formats: Vec<String>,
        /// Used only when converting .pth weights; never used for inference.
        #[arg(long, default_value = "python")]
        python: String,
    },
    Convert {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "python")]
        python: String,
    },
}
fn convert(input: &Path, out: &Path, python: &str) -> Result<()> {
    let status=std::process::Command::new(python).arg("-c").arg(include_str!("../tools/export_weights.py")).arg("--input").arg(input).arg("--output").arg(out).status().context("Launch checkpoint exporter; install Python, torch and safetensors, or supply .safetensors")?;
    ensure!(
        status.success(),
        "Checkpoint conversion failed; no inference was started"
    );
    Ok(())
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    progress::configure(!cli.no_progress);
    if let Command::Convert { model, out, python } = &cli.command {
        return convert(model, out, python);
    }
    ensure!(
        cli.encoder_batch > 0
            && cli.pixel_batch > 0
            && cli.inflight_mib > 0
            && cli.inflight_mib <= 16384
            && cli.score_budget_mib > 0
            && cli.score_budget_mib <= 16384,
        "Batch sizes and memory budget must be positive (budget <=16384 MiB)"
    );
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    let executable = std::env::current_exe()?;
    ensure!(
        executable.is_file(),
        "Cannot select process-local runtime storage"
    );
    // Pinned CubeCL's throughput storage lacks a memory-only switch. An existing
    // file cannot be a directory; upstream falls back to RAM before opening a DB.
    config.environment.path = CacheConfig::Directory(executable);
    config.compilation.cache = false;
    config.autotune.disable_cache = true;
    config.throughput.disable_cache = false;
    config.environment.records.level = cubecl::records::RecordLevel::Off;
    config.environment.records.keep_sessions = Some(0);
    ensure!(
        CubeClRuntimeConfig::try_set(config),
        "Runtime already initialized"
    );
    let device = if cli.cpu {
        Device::flex()
    } else if cli.api == "cuda" {
        #[cfg(feature = "cuda")]
        {
            Device::cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("Build with --features cuda for CUDA; Vulkan is the cross-vendor default")
        }
    } else {
        use burn::tensor::wgpu::WgpuBackend;
        Device::wgpu_options()
            .device_kind(if cli.integrated {
                DeviceKind::IntegratedGpu(0)
            } else {
                DeviceKind::DefaultDevice
            })
            .graphics_api(match cli.api.as_str() {
                "dx12" => WgpuBackend::Dx12,
                "metal" => WgpuBackend::Metal,
                "vulkan" => WgpuBackend::Vulkan,
                _ => {
                    #[cfg(any(target_os = "windows", target_os = "linux"))]
                    {
                        WgpuBackend::Vulkan
                    }
                    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
                    {
                        WgpuBackend::Auto
                    }
                }
            })
            .init()
            .map_err(|e| anyhow::anyhow!("Device initialization: {e:?}"))?
    };
    let dtype = match cli.precision.as_str() {
        "bf16" => FloatDType::BF16,
        "fp16" => FloatDType::F16,
        "fp32" => FloatDType::F32,
        _ => {
            if cli.cpu {
                FloatDType::F32
            } else if device.supports_dtype(FloatDType::BF16) {
                FloatDType::BF16
            } else if device.supports_dtype(FloatDType::F16) {
                FloatDType::F16
            } else {
                FloatDType::F32
            }
        }
    };
    eprintln!(
        "lino {} | {:?} | {dtype:?}",
        env!("LINO_REVISION"),
        device.identity()
    );
    let Command::Infer {
        model,
        data,
        out,
        mode,
        images,
        batch,
        prefix,
        seed,
        normalization,
        max_resolution,
        formats,
        python,
    } = cli.command
    else {
        unreachable!()
    };
    ensure!(
        images == -1 || images > 0,
        "--images must be -1 or positive"
    );
    ensure!(
        max_resolution >= 512 && max_resolution % 512 == 0,
        "--max-resolution must be a positive multiple of 512"
    );
    let checkpoint = if model.extension().and_then(|s| s.to_str()) == Some("pth") {
        let p = model.with_extension("lino-rs.safetensors");
        convert(&model, &p, &python)?;
        p
    } else {
        model.clone()
    };
    let pbr = matches!(mode, Mode::Pbr);
    let mut ops = ops::Ops::load(&checkpoint, device, dtype, cli.score_budget_mib, pbr)?;
    ops.reference_kernels = cli.reference_kernels;
    ops.inflight_mib = cli.inflight_mib;
    let net = network::Network {
        ops,
        pbr,
        encoder_batch: cli.encoder_batch,
        verbose: cli.verbose,
    };
    let mut objects = if batch {
        std::fs::read_dir(&data)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect::<Vec<_>>()
    } else {
        vec![data]
    };
    objects.sort();
    ensure!(!objects.is_empty(), "No input objects");
    let mut rng = Rng::new(seed);
    let total_views = objects.len();
    for (view_index, object) in objects.into_iter().enumerate() {
        let dir = if batch {
            out.join(object.file_name().unwrap())
        } else {
            out.clone()
        };
        std::fs::create_dir_all(&dir)?;
        let view_name = object
            .file_name()
            .unwrap_or(object.as_os_str())
            .to_string_lossy();
        let view_progress = progress::view(&view_name, view_index + 1, total_views);
        let start = Instant::now();
        let loading = progress::phase("load and normalize images", 1, "view");
        let input = Input::load(
            &object,
            images,
            &prefix,
            max_resolution,
            dtype,
            &normalization,
            &mut rng,
        )?;
        drop(loading);
        let input_seconds = start.elapsed().as_secs_f64();
        net.ops.reset_timings();
        let inference = Instant::now();
        let n = input.images.len();
        let grid = input.size / 512;
        let mut maps = vec![Map::new("normal", 3, input.size)];
        if pbr {
            maps.extend([
                Map::new("diffuse", 3, input.size),
                Map::new("roughness", 1, input.size),
                Map::new("metallic", 1, input.size),
            ]);
        }
        progress::message(format!(
            "{}: {n} lights, {}px, {} interleaved blocks",
            object.display(),
            input.size,
            grid * grid
        ));
        let mut profile = Vec::new();
        for block in 0..grid * grid {
            view_progress.block(block + 1, grid * grid);
            let began = Instant::now();
            let (rgb, mask) = input.block(block, dtype);
            let mut ids: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.then_some(i))
                .collect();
            if ids.is_empty() {
                continue;
            }
            let masked: Vec<_> = rgb
                .iter()
                .map(|im| {
                    let mut v = im.range(0..im.len());
                    for c in 0..3 {
                        for (i, m) in mask.iter().enumerate() {
                            if !*m {
                                v[c * 512 * 512 + i] = 0.;
                            }
                        }
                    }
                    Packed::new(v, dtype)
                })
                .collect();
            let features = net.encode(&masked);
            drop(masked);
            let encode_seconds = began.elapsed().as_secs_f64();
            rng.shuffle(&mut ids);
            rng.shuffle(&mut ids);
            let groups = ids.len() / cli.pixel_batch + 1;
            let base = ids.len() / groups;
            let extra = ids.len() % groups;
            let mut at = 0;
            let decoding = progress::phase("decode sampled pixels", ids.len(), "pixels");
            let mut prefetched = None;
            for group in 0..groups {
                let count = base + usize::from(group < extra);
                let set = &ids[at..at + count];
                at += count;
                if count == 0 {
                    continue;
                }
                let (colors, f) = prefetched
                    .take()
                    .unwrap_or_else(|| gather_inputs(&rgb, &features, set, dtype));
                let predictions = net.decode(
                    upload(colors, [count, n, 3], &net.ops.device, dtype),
                    f.into_tensor([count, n, 256], &net.ops.device),
                );
                // Submit the current GPU work, then gather the next CPU batch while
                // it runs. Only host buffers are double-buffered, not dense GPU maps.
                net.ops.boundary();
                if group + 1 < groups {
                    let next_count = base + usize::from(group + 1 < extra);
                    if next_count > 0 {
                        prefetched = Some(gather_inputs(
                            &rgb,
                            &features,
                            &ids[at..at + next_count],
                            dtype,
                        ));
                    }
                }
                for (name, tensor) in predictions {
                    let values = read(tensor);
                    let map = maps.iter_mut().find(|m| m.name == name).unwrap();
                    for (i, &pixel) in set.iter().enumerate() {
                        let y = (pixel / 512) * grid + block / grid;
                        let x = (pixel % 512) * grid + block % grid;
                        let dst = (y * input.size + x) * map.channels;
                        map.values[dst..dst + map.channels]
                            .copy_from_slice(&values[i * map.channels..(i + 1) * map.channels]);
                    }
                }
                // Readback above already completes the predictions. No new GPU wait.
                decoding.set(at);
            }
            drop(decoding);
            drop(features);
            net.ops.trim_after_readback();
            let elapsed = began.elapsed().as_secs_f64();
            if cli.verbose {
                progress::message(format!(
                    "Block {}/{}: encoder {encode_seconds:.2}s, total {elapsed:.2}s",
                    block + 1,
                    grid * grid
                ));
            }
            profile.push(serde_json::json!({"block":block,"encoder_seconds":encode_seconds,"seconds":elapsed,"foreground":ids.len()}));
        }
        let inference_seconds = inference.elapsed().as_secs_f64();
        let saving = progress::phase("save output maps", maps.len(), "maps");
        for (index, map) in maps.iter().enumerate() {
            output::save(map, &input, &dir, &formats)?;
            saving.set(index + 1);
        }
        let metadata = serde_json::json!({"revision":env!("LINO_REVISION"),"upstream":"4ddb4113d0196b853978d2cc72de582611b01ebd","model":model,"mode":format!("{mode:?}"),"precision":format!("{dtype:?}"),"device":format!("{:?}",net.ops.device.identity()),"images":n,"size":input.size,"files":input.files,"roi":input.roi,"normalization":normalization,"seed":seed,"pixel_batch":cli.pixel_batch,"encoder_batch":cli.encoder_batch,"score_budget_mib":cli.score_budget_mib,"inflight_mib":cli.inflight_mib,"reference_kernels":cli.reference_kernels,"attention_dispatch_totals":fast_attention::stats(),"operation_timings":net.ops.timing_report(),"input_seconds":input_seconds,"inference_seconds":inference_seconds,"seconds":start.elapsed().as_secs_f64(),"material_mapping":"clamp(raw,-1,1)*0.5+0.5; diffuse is upstream baseColor","normal_axes":"upstream xyz","formats":formats});
        std::fs::write(
            dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        if cli.verbose {
            std::fs::write(
                dir.join("profile.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"blocks":profile,"operations":net.ops.timing_report(),"timing_kind":"inclusive CPU wall time, includes waits when encountered; nested entries overlap"}),
                )?,
            )?;
        }
        drop(saving);
        view_progress.finish();
        println!(
            "Saved {} ({inference_seconds:.2}s inference)",
            dir.display()
        );
    }
    Ok(())
}

fn gather_inputs(
    rgb: &[Packed],
    features: &[Packed],
    pixels: &[usize],
    dtype: FloatDType,
) -> (Vec<f32>, Packed) {
    let n = rgb.len();
    let mut packed = Packed::empty(dtype, pixels.len() * n * 256);
    let mut colors = Vec::with_capacity(pixels.len() * n * 3);
    for &pixel in pixels {
        for light in 0..n {
            packed.extend_range(&features[light], pixel * 256..(pixel + 1) * 256);
            for c in 0..3 {
                colors.push(rgb[light].get(c * 512 * 512 + pixel));
            }
        }
    }
    (colors, packed)
}

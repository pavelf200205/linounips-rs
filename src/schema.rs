//! Exact released architecture: ViT-S/8, four alternating-attention stages.
use std::collections::BTreeMap;
type Schema = BTreeMap<String, Vec<usize>>;
fn add(s: &mut Schema, k: &str, v: &[usize]) {
    s.insert(k.into(), v.to_vec());
}
fn linear(s: &mut Schema, p: &str, i: usize, o: usize, b: bool) {
    add(s, &format!("{p}.weight"), &[o, i]);
    if b {
        add(s, &format!("{p}.bias"), &[o]);
    }
}
fn norm(s: &mut Schema, p: &str, c: usize) {
    add(s, &format!("{p}.weight"), &[c]);
    add(s, &format!("{p}.bias"), &[c]);
}
fn conv(s: &mut Schema, p: &str, i: usize, o: usize, k: usize, b: bool) {
    add(s, &format!("{p}.weight"), &[o, i, k, k]);
    if b {
        add(s, &format!("{p}.bias"), &[o]);
    }
}
fn block(s: &mut Schema, p: &str, qkn: bool) {
    for n in ["norm1", "norm2"] {
        norm(s, &format!("{p}.{n}"), 384)
    }
    linear(s, &format!("{p}.attn.qkv"), 384, 1152, true);
    linear(s, &format!("{p}.attn.proj"), 384, 384, true);
    linear(s, &format!("{p}.mlp.fc1"), 384, 1536, true);
    linear(s, &format!("{p}.mlp.fc2"), 1536, 384, true);
    for n in ["ls1", "ls2"] {
        add(s, &format!("{p}.{n}.gamma"), &[384]);
    }
    if qkn {
        norm(s, &format!("{p}.attn.q_norm"), 64);
        norm(s, &format!("{p}.attn.k_norm"), 64);
    }
}
fn mab(s: &mut Schema, p: &str, i: usize, o: usize, ff: usize, ln: bool) {
    for n in ["fc_q", "fc_k", "fc_v"] {
        linear(s, &format!("{p}.{n}"), i, o, false)
    }
    linear(s, &format!("{p}.fc_o1"), o, ff, false);
    linear(s, &format!("{p}.fc_o2"), ff, o, false);
    if ln {
        norm(s, &format!("{p}.ln0"), i);
        norm(s, &format!("{p}.ln1"), o);
    }
}
pub fn expected(pbr: bool) -> Schema {
    let mut s = Schema::new();
    let a = "image_encoder.backbone.aggregator";
    let d = format!("{a}.patch_embed");
    conv(&mut s, &format!("{d}.patch_embed.proj"), 3, 384, 8, true);
    add(&mut s, &format!("{d}.cls_token"), &[1, 1, 384]);
    add(&mut s, &format!("{d}.pos_embed"), &[1, 1025, 384]);
    add(&mut s, &format!("{d}.register_tokens"), &[1, 4, 384]);
    norm(&mut s, &format!("{d}.norm"), 384);
    for i in 0..12 {
        block(&mut s, &format!("{d}.blocks.{i}"), false)
    }
    for n in [
        "frame_blocks",
        "light_axis_1_blocks",
        "global_blocks",
        "light_axis_2_blocks",
    ] {
        for i in 0..4 {
            block(&mut s, &format!("{a}.{n}.{i}"), true)
        }
    }
    let f = "image_encoder.fusion";
    norm(&mut s, &format!("{f}.norm"), 1536);
    for (i, c) in [256, 512, 1024, 1024].into_iter().enumerate() {
        conv(&mut s, &format!("{f}.projects.{i}.0"), 384, c, 1, true);
        conv(
            &mut s,
            &format!("{f}.scratch.layer{}_rn.0", i + 1),
            c,
            256,
            3,
            false,
        );
        let k = if i == 2 { 1 } else { 2 };
        conv(&mut s, &format!("{f}.resize_layers.{i}.0"), c, c, k, true);
        if i == 0 {
            conv(&mut s, &format!("{f}.resize_layers.0.2"), c, c, 2, true);
        }
    }
    for i in 1..=4 {
        let p = format!("{f}.scratch.refinenet{i}");
        conv(&mut s, &format!("{p}.out_conv"), 256, 256, 1, true);
        for unit in 1..=2 {
            if i == 4 && unit == 1 {
                continue;
            }
            for c in 1..=2 {
                conv(
                    &mut s,
                    &format!("{p}.resConfUnit{unit}.conv{c}"),
                    256,
                    256,
                    3,
                    true,
                );
            }
        }
    }
    conv(
        &mut s,
        &format!("{f}.scratch.output_conv1"),
        256,
        256,
        3,
        true,
    );
    linear(&mut s, "img_embedding.0", 3, 32, true);
    linear(&mut s, "img_embedding.2", 32, 256, true);
    for i in 0..2 {
        mab(
            &mut s,
            &format!("glc_upsample.comm.enc.{i}.mab"),
            256,
            256,
            1024,
            true,
        );
        mab(
            &mut s,
            &format!("regressor.comm.enc.{i}.mab"),
            384,
            384,
            1024,
            true,
        );
    }
    for i in 0..3 {
        mab(
            &mut s,
            &format!("glc_aggregation.aggregation.enc.{i}.mab"),
            if i == 0 { 256 } else { 384 },
            384,
            1024,
            true,
        );
    }
    add(&mut s, "glc_aggregation.aggregation.dec.0.S", &[1, 1, 384]);
    mab(
        &mut s,
        "glc_aggregation.aggregation.dec.0.mab",
        384,
        384,
        512,
        false,
    );
    for (name, c) in [("normal", 3), ("base", 3), ("rough", 1), ("metal", 1)] {
        if name != "normal" && !pbr {
            continue;
        }
        let p = format!("regressor.prediction_{name}");
        linear(&mut s, &format!("{p}.regression.0"), 384, 192, true);
        linear(&mut s, &format!("{p}.out_layer"), 192, c, true);
    }
    s
}

struct GlobalUniforms {
    timestamp: f32,
    resolution: vec2<f32>,
    _pad0: vec3<f32>,
}; 

@group(0) @binding(0)
var<uniform> globalUniforms: GlobalUniforms;

@group(1) @binding(1) var canvasSampler: sampler;
@group(1) @binding(2) var canvasTexture: texture_2d<f32>;

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return fract(sin(q) * 43758.5453123);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = dot(hash22(i + vec2<f32>(0.0, 0.0)) - 0.5, f - vec2<f32>(0.0, 0.0));
    let b = dot(hash22(i + vec2<f32>(1.0, 0.0)) - 0.5, f - vec2<f32>(1.0, 0.0));
    let c = dot(hash22(i + vec2<f32>(0.0, 1.0)) - 0.5, f - vec2<f32>(0.0, 1.0));
    let d = dot(hash22(i + vec2<f32>(1.0, 1.0)) - 0.5, f - vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var p0 = p;
    var value = 0.0;
    var amplitude = 0.5;

    for (var i = 0; i < 5; i = i + 1) {
        value = value + amplitude * noise2(p0);
        p0 = p0 * 2.03 + vec2<f32>(13.7, 7.9);
        amplitude = amplitude * 0.5;
    }

    return value;
}

fn baseNoise(uv: vec2<f32>) -> f32 {
    let n1 = fbm(uv * 7.0 + vec2<f32>(3.1, 5.7));
    let n2 = fbm(uv * 13.0 + vec2<f32>(11.3, 2.9));
    let layered = (n1 * 0.5 + 0.5) * 0.65 + (n2 * 0.5 + 0.5) * 0.35;

    return layered * 0.06;
}

fn remap(value: f32, fromMin: f32, fromMax: f32, toMin: f32, toMax: f32) -> f32 {
    let t = (value - fromMin) / (fromMax - fromMin);
    return mix(toMin, toMax, t);
}

fn my_noise(uv: vec2<f32>, t: f32) -> f32 {
    let base = uv * 4.0;
    let warp1_3 = vec2<f32>(
        fbm(base + vec2<f32>(1.7 + t * 1.2, 9.2 - t * 0.7)),
        fbm(base + vec2<f32>(8.3 - t * 0.9, 2.8 + t * 1.1)),
        // fbm(base + vec2<f32>(5.1 + t * 0.5, 1.3 - t * 0.3))
    );
    let warp1 = warp1_3.xy;
    var x = fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3));
    // let warp2 = vec3<f32>(
    //     fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3) + t * 2),
    //     fbm(base + 2.5 * warp1 + vec2<f32>(2.4, 7.6)),
    //     fbm(base + 2.5 * warp1 + vec2<f32>(3.7, 4.2))
    // );
    // let warp2 = fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3));
    let n01 = clamp(remap(x, -0.35, 0.35, 0.0, 1.0), 0.0, 1.0);
    return 1.0 - pow(1.0 - n01, 2.2);
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res: vec2<f32> = globalUniforms.resolution;
    let uv: vec2<f32> = pos.xy / res;
    let t: f32 = globalUniforms.timestamp * 0.0001;

    return warppedSample(uv, t);
}

fn warppedSample(uv: vec2<f32>, t: f32) -> vec4<f32> {
    let base = uv * 4.0;
    let warp1 = vec2<f32>(
        fbm(base + vec2<f32>(1.7 + t * 1.2, 9.2 - t * 0.7)),
        fbm(base + vec2<f32>(8.3 - t * 0.9, 2.8 + t * 1.1))
    );
    let warp2 = vec2<f32>(
        fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3)),
        fbm(base + 2.5 * warp1 + vec2<f32>(2.4, 7.6)),
        // fbm(base + 2.5 * warp1 + vec2<f32>(3.7, 4.2))
    );
    let warpedUv = uv + warp2.xy * 0.1;
    _ = t;
    return textureSample(canvasTexture, canvasSampler, warpedUv);
}
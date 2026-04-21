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

    for (var i = 0; i < 4; i = i + 1) {
        value = value + amplitude * noise2(p0);
        p0 = p0 * 2.03 + vec2<f32>(13.7, 7.9);
        amplitude = amplitude * 0.5;
    }

    return value;
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res: vec2<f32> = globalUniforms.resolution;
    let uv: vec2<f32> = pos.xy / res;
    let t: f32 = globalUniforms.timestamp * 0.0001;

    let base = uv * 4.0;
    let warp1 = vec2<f32>(
        fbm(base + vec2<f32>(1.7 + t * 1.2, 9.2 - t * 0.7)),
        fbm(base + vec2<f32>(8.3 - t * 0.9, 2.8 + t * 1.1))
    );
    let warp2 = vec2<f32>(
        fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3)),
        fbm(base + 2.5 * warp1 + vec2<f32>(2.4, 7.6))
    );

    // The source texture carries signal in alpha only.
    let alphaBase = textureSample(canvasTexture, canvasSampler, uv).a;
    let warpAmount = 0.02 + alphaBase * 0.12;
    let warpedUv = fract(uv + (warp2 - 0.5) * warpAmount);
    let warpedAlpha = textureSample(canvasTexture, canvasSampler, warpedUv).a;

    return vec4<f32>(vec3<f32>(warpedAlpha), 1.0);
}
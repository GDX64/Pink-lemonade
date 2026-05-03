struct GlobalUniforms {
    timestamp: f32,
    resolution: vec2<f32>,
    _pad0: vec3<f32>,
}; 

@group(0) @binding(0)
var<uniform> globalUniforms: GlobalUniforms;

@group(1) @binding(1) var canvasSampler: sampler;
@group(1) @binding(2) var canvasTexture: texture_2d<f32>;

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res: vec2<f32> = globalUniforms.resolution;
    let uv: vec2<f32> = pos.xy / res;
    let t: f32 = globalUniforms.timestamp * 0.0001;

    // The source texture carries signal in alpha only.
    // return textureSample(canvasTexture, canvasSampler, uv);
    var alphaBase = sample(uv);
    let out = mapColor(alphaBase);
    return vec4<f32>(out.rgb, 1.0);
    // let warpLen = pow((1.0 - length(warp2)), 1.0);
    // return vec4<f32>(warpLen, warpLen, warpLen, 1.0);
    // return vec4<f32>(vec3<f32>(alphaBase), 1.0);
    // let out = bookOfShadersWarpingOutput(uv, t * 10.0);
    // return vec4<f32>(vec3<f32>(alphaBase), 1.0);
    // return vec4<f32>(out.rgb, 1.0);
}

fn mapColor(value: f32) -> vec3<f32> {
    let x = clamp(value, 0.0, 1.0);
    let c0 = vec3<f32>(0.00, 0.00, 0.00);
    let c1 = vec3<f32>(0.10, 0.45, 0.75);
    let c2 = vec3<f32>(0.95, 0.80, 0.35);
    let c3 = vec3<f32>(0.98, 0.30, 0.15);

    let t01 = smoothstep(0.00, 0.35, x);
    let t12 = smoothstep(0.35, 0.70, x);
    let t23 = smoothstep(0.70, 1.00, x);

    let col01 = mix(c0, c1, t01);
    let col12 = mix(c1, c2, t12);
    let col23 = mix(c2, c3, t23);
    let lowMid = mix(col01, col12, step(0.35, x));
    return mix(lowMid, col23, step(0.70, x));
}

fn decodePackedRgb01(s: vec4<f32>) -> f32 {
    let r = u32(round(clamp(s.r, 0.0, 1.0) * 255.0));
    let g = u32(round(clamp(s.g, 0.0, 1.0) * 255.0));
    let b = u32(round(clamp(s.b, 0.0, 1.0) * 255.0));
    let val = (r << 16u) | (g << 8u) | b;
    return f32(val) / 16777215.0;
}

fn catmullRomWeights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;

    let w0 = -0.5 * t + t2 - 0.5 * t3;
    let w1 = 1.0 - 2.5 * t2 + 1.5 * t3;
    let w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3;
    let w3 = -0.5 * t2 + 0.5 * t3;
    return vec4<f32>(w0, w1, w2, w3);
}

fn bicubicSamplePackedScalar(uv: vec2<f32>) -> f32 {
    let texSizeU = textureDimensions(canvasTexture, 0);
    let texSizeI = vec2<i32>(texSizeU);
    let texSize = vec2<f32>(texSizeU);

    let pos = uv * texSize - vec2<f32>(0.5, 0.5);
    let base = vec2<i32>(floor(pos));
    let f = fract(pos);

    let wxV = catmullRomWeights(f.x);
    let wyV = catmullRomWeights(f.y);
    let wx = array<f32, 4>(wxV.x, wxV.y, wxV.z, wxV.w);
    let wy = array<f32, 4>(wyV.x, wyV.y, wyV.z, wyV.w);

    var acc = 0.0;
    for (var jy = 0; jy < 4; jy = jy + 1) {
        let y = clamp(base.y + jy - 1, 0, texSizeI.y - 1);
        for (var ix = 0; ix < 4; ix = ix + 1) {
            let x = clamp(base.x + ix - 1, 0, texSizeI.x - 1);
            let v = textureLoad(canvasTexture, vec2<i32>(x, y), 0).r;
            acc = acc + wx[ix] * wy[jy] * v;
        }
    }

    return clamp(acc, 0.0, 1.0);
}

// Fast bicubic approximation using 4 bilinear taps.
fn fastBicubicSamplePackedScalar(uv: vec2<f32>) -> f32 {
    let texSize = vec2<f32>(textureDimensions(canvasTexture, 0));
    let texel = vec2<f32>(1.0, 1.0) / texSize;

    let p = uv * texSize - vec2<f32>(0.5, 0.5);
    let f = fract(p);
    let f2 = f * f;
    let f3 = f2 * f;

    // Catmull-Rom pairwise weights, grouped into two bilinear taps per axis.
    let w0 = -0.5 * f + f2 - 0.5 * f3;
    let w1 = 1.0 - 2.5 * f2 + 1.5 * f3;
    let w2 = 0.5 * f + 2.0 * f2 - 1.5 * f3;
    let w3 = -0.5 * f2 + 0.5 * f3;

    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let eps = vec2<f32>(1e-6, 1e-6);

    let h0 = (w1 / (g0 + eps)) - 0.5;
    let h1 = (w3 / (g1 + eps)) + 1.5;

    let base = (floor(p) + vec2<f32>(0.5, 0.5)) * texel;
    let uv00 = base + vec2<f32>(h0.x, h0.y) * texel;
    let uv10 = base + vec2<f32>(h1.x, h0.y) * texel;
    let uv01 = base + vec2<f32>(h0.x, h1.y) * texel;
    let uv11 = base + vec2<f32>(h1.x, h1.y) * texel;

    let s00 = decodePackedRgb01(textureSample(canvasTexture, canvasSampler, uv00));
    let s10 = decodePackedRgb01(textureSample(canvasTexture, canvasSampler, uv10));
    let s01 = decodePackedRgb01(textureSample(canvasTexture, canvasSampler, uv01));
    let s11 = decodePackedRgb01(textureSample(canvasTexture, canvasSampler, uv11));

    let gx0 = g0.x;
    let gx1 = g1.x;
    let gy0 = g0.y;
    let gy1 = g1.y;

    let top = s00 * gx0 + s10 * gx1;
    let bottom = s01 * gx0 + s11 * gx1;
    let out = top * gy0 + bottom * gy1;
    return clamp(out, 0.0, 1.0);
}

fn sample(uv: vec2<f32>) -> f32 {
    return bicubicSamplePackedScalar(uv);
    // return textureSample(canvasTexture, canvasSampler, uv).r;
}


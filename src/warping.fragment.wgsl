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

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res: vec2<f32> = globalUniforms.resolution;
    let uv: vec2<f32> = pos.xy / res;
    let t: f32 = globalUniforms.timestamp * 0.0001;

    // The source texture carries signal in alpha only.
    // return textureSample(canvasTexture, canvasSampler, uv);
    var alphaBase = warppedSample(uv, t);
    let out = mapColor(alphaBase);
    return vec4<f32>(out.rgb, 1.0);
    // let warpLen = pow((1.0 - length(warp2)), 1.0);
    // return vec4<f32>(warpLen, warpLen, warpLen, 1.0);
    // return vec4<f32>(vec3<f32>(alphaBase), 1.0);
    // let out = bookOfShadersWarpingOutput(uv, t * 10.0);
    // return vec4<f32>(vec3<f32>(alphaBase), 1.0);
    // return vec4<f32>(out.rgb, 1.0);
}

fn remap(value: f32, fromMin: f32, fromMax: f32, toMin: f32, toMax: f32) -> f32 {
    let t = (value - fromMin) / (fromMax - fromMin);
    return mix(toMin, toMax, t);
}

fn mapColor(value: f32) -> vec3<f32> {
    let x = clamp(value, 0.0, 1.0);
    let c0 = vec3<f32>(0.03, 0.05, 0.20);
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

fn warppedSample(uv: vec2<f32>, t: f32) -> f32 {
    // let base = uv * 4.0;
    // let warp1 = vec2<f32>(
    //     fbm(base + vec2<f32>(1.7 + t * 1.2, 9.2 - t * 0.7)),
    //     fbm(base + vec2<f32>(8.3 - t * 0.9, 2.8 + t * 1.1))
    // );
    // let warp2 = vec2<f32>(
    //     fbm(base + 2.5 * warp1 + vec2<f32>(5.1, 1.3)),
    //     fbm(base + 2.5 * warp1 + vec2<f32>(2.4, 7.6)),
    //     // fbm(base + 2.5 * warp1 + vec2<f32>(3.7, 4.2))
    // );
    // let warpedUv = uv + warp2.xy * 0.05;
    _ = t;
    // return fastBicubicSamplePackedScalar(uv);
    return bicubicSamplePackedScalar(uv);
    // return decodePackedRgb01(textureSample(canvasTexture, canvasSampler, uv));
}


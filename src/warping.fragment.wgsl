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

fn bosRandom(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn bosNoise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);

    let a = bosRandom(i);
    let b = bosRandom(i + vec2<f32>(1.0, 0.0));
    let c = bosRandom(i + vec2<f32>(0.0, 1.0));
    let d = bosRandom(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
        + (c - a) * u.y * (1.0 - u.x)
        + (d - b) * u.x * u.y;
}

fn bosFbm(st: vec2<f32>) -> f32 {
    var p = st;
    var v = 0.0;
    var a = 0.5;
    let shift = vec2<f32>(100.0, 100.0);
    let c = cos(0.5);
    let s = sin(0.5);
    let rot = mat2x2<f32>(c, s, -s, c);

    for (var i = 0; i < 5; i = i + 1) {
        v = v + a * bosNoise(p);
        p = rot * p * 2.0 + shift;
        a = a * 0.5;
    }

    return v;
}

fn bookOfShadersWarpingOutput(uv: vec2<f32>, time: f32) -> vec4<f32> {
    let st = uv * 3.0;

    var q = vec2<f32>(0.0, 0.0);
    q.x = bosFbm(st + 0.00 * time);
    q.y = bosFbm(st + vec2<f32>(1.0, 1.0));

    var r = vec2<f32>(0.0, 0.0);
    r.x = bosFbm(st + 1.0 * q + vec2<f32>(1.7, 9.2) + vec2<f32>(0.15 * time, 0.15 * time));
    r.y = bosFbm(st + 1.0 * q + vec2<f32>(8.3, 2.8) + vec2<f32>(0.126 * time, 0.126 * time));

    let f = bosFbm(st + r);

    var color = mix(
        vec3<f32>(0.101961, 0.619608, 0.666667),
        vec3<f32>(0.666667, 0.666667, 0.498039),
        clamp((f * f) * 4.0, 0.0, 1.0)
    );

    color = mix(
        color,
        vec3<f32>(0.0, 0.0, 0.164706),
        clamp(length(q), 0.0, 1.0)
    );

    color = mix(
        color,
        vec3<f32>(0.666667, 1.0, 1.0),
        clamp(abs(r.x), 0.0, 1.0)
    );

    let shaded = (f * f * f + 0.6 * f * f + 0.5 * f) * color;
    return vec4<f32>(shaded, 1.0);
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

fn warppedSample(uv: vec2<f32>, t: f32) -> f32 {
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
    let s = textureSample(canvasTexture, canvasSampler, uv + warp2.xy * 0.1);
    let r = i32(s.r * 255.0);
    let g = i32(s.g * 255.0);
    let b = i32(s.b * 255.0);
    let val = (r << 16) + (g << 8) + b;
    // return s.g;
    return f32(val) / 16777215.0;
}

fn blury_sample(uv: vec2<f32>) -> f32 {
    const BLUR_PX: f32 = 0.0;
    let texSize: vec2<f32> = vec2<f32>(textureDimensions(canvasTexture, 0));
    let texel: vec2<f32> = vec2<f32>(1.0, 1.0) / texSize;
    let o = texel * BLUR_PX;

    let s00 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(-o.x, -o.y)).a;
    let s10 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(0.0, -o.y)).a;
    let s20 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(o.x, -o.y)).a;

    let s01 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(-o.x, 0.0)).a;
    let s11 = textureSample(canvasTexture, canvasSampler, uv).a;
    let s21 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(o.x, 0.0)).a;

    let s02 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(-o.x, o.y)).a;
    let s12 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(0.0, o.y)).a;
    let s22 = textureSample(canvasTexture, canvasSampler, uv + vec2<f32>(o.x, o.y)).a;

    let sum = s00 * 1.0 + s10 * 2.0 + s20 * 1.0 +
        s01 * 2.0 + s11 * 4.0 + s21 * 2.0 +
        s02 * 1.0 + s12 * 2.0 + s22 * 1.0;

    return sum / 16.0;
}
fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0; i < 5; i = i + 1) {
        value = value + amplitude * noise2(p * frequency);
        frequency = frequency * 2.0;
        amplitude = amplitude * 0.5;
    }

    return value;
}

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let time: f32 = globalUniforms.timestamp * 0.001;

    // Tile flames so each rectangle gets multiple tongues regardless of size.
    let uv = fract(fragCoord.xy / vec2<f32>(400.0, 400.0));
    let height = 1.0 - uv.y;

    let baseNoise = fbm(vec2<f32>(uv.x * 3.5, height * 4.0 - time * 1.8));
    let drift = (baseNoise - 0.5) * (0.55 - 0.35 * height);
    // Squash X into quarter scale so the flame reads about 4x wider than tall.
    let flameX = ((uv.x * 2.0 - 1.0) + drift) * 0.25;

    let width = mix(0.6, 0.1, height);
    let silhouette = 1.0 - smoothstep(width, width + 0.15, abs(flameX));

    let flickerNoise = fbm(vec2<f32>(flameX * 20.0 + time * 0.7, height * 7.5 - time * 4.2));
    let flicker = smoothstep(0.22, 0.95, flickerNoise);
    let tipFade = 1.0 - smoothstep(0.75, 1.0, height);
    let intensity = silhouette * tipFade * (0.5 + 0.5 * flicker);

    let dark = vec3<f32>(0.03, 0.01, 0.01);
    let ember = vec3<f32>(0.95, 0.23, 0.03);
    let hot = vec3<f32>(1.0, 0.86, 0.26);

    let warmMix = clamp(intensity * 1.35, 0.0, 1.0);
    let hotMix = clamp(intensity * intensity * 2.0, 0.0, 1.0);
    let color = mix(mix(dark, ember, warmMix), hot, hotMix);

    return vec4<f32>(color, 1.0);
}

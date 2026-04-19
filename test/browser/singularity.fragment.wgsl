
struct SingularityUniforms {
    speed: f32,
    intensity: f32,
    size: f32,
    waveStrength: f32,
    colorShift: f32,
    u_resolution_x: f32,
    u_resolution_y: f32,
    _pad0: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> singularityUniforms: SingularityUniforms;

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    var i: f32 = 0.2 * singularityUniforms.speed;

    let resolution = vec2<f32>(singularityUniforms.u_resolution_x, singularityUniforms.u_resolution_y);
    let F = vec2<f32>(fragCoord.x, resolution.y - fragCoord.y);

    let p = (F + F - resolution) / resolution.y / (0.7 * singularityUniforms.size);
    let d = vec2<f32>(-1.0, 1.0);
    let b = p - i * d;

    let bLen2 = max(dot(b, b), 1e-5);
    let k = 0.1 + i / bLen2;

    // GLSL uses row-vector * mat2, so we keep the equivalent arithmetic directly.
    let c = vec2<f32>(
        p.x + p.y,
        (p.y - p.x) / k,
    );

    let a = max(dot(c, c), 1e-5);
    let t = globalUniforms.timestamp * 0.001;
    let q = 0.5 * log(a) + t * i * singularityUniforms.speed;

    let m0 = cos(q + 0.0);
    let m1 = cos(q + 33.0);
    let m2 = cos(q + 11.0);
    let m3 = cos(q + 0.0);

    var v = vec2<f32>(
        c.x * m0 + c.y * m1,
        c.x * m2 + c.y * m3,
    ) / max(i, 1e-5);

    var w = vec2<f32>(0.0, 0.0);

    for (var j = 0; j < 9; j = j + 1) {
        i = i + 1.0;
        w = w + (vec2<f32>(1.0, 1.0) + sin(v * singularityUniforms.waveStrength));
        v = v + 0.7 * sin(vec2<f32>(v.y, v.x) * i + t * singularityUniforms.speed) / i + 0.5;
    }

    i = length(sin(v / 0.3) * 0.4 + c * (3.0 + d));

    let colorGrad = vec4<f32>(0.6, -0.4, -1.0, 0.0) * singularityUniforms.colorShift;
    let wxyyx = vec4<f32>(w.x, w.y, w.y, w.x);

    let denom = wxyyx *
        (2.0 + i * i / 4.0 - i) *
        (0.5 + 1.0 / a) *
        (0.03 + abs(length(p) - 0.7));

    let energy = exp(c.x * colorGrad) / max(denom, vec4<f32>(1e-5));
    let O = vec4<f32>(1.0, 1.0, 1.0, 1.0) - exp(-energy * singularityUniforms.intensity);

    return O;
}

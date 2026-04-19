struct PulseUniforms {
    speed: f32,
    intensity: f32,
    height: f32,
    turbulence: f32,
    colorShift: f32,
    baseColor: vec4<f32>,
    u_resolution_x: f32,
    u_resolution_y: f32,
    _pad0: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> pulseUniforms: PulseUniforms;

fn tanhApprox(x: vec4<f32>) -> vec4<f32> {
    let x2 = x * x;
    return x * (vec4<f32>(3.0) + x2) / (vec4<f32>(3.0) + 3.0 * x2);
}

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let t: f32 = globalUniforms.timestamp * 0.001 * pulseUniforms.speed;
    let resolution = vec2<f32>(pulseUniforms.u_resolution_x, pulseUniforms.u_resolution_y);
    let I = vec2<f32>(fragCoord.x, resolution.y - fragCoord.y);

    var O = vec4<f32>(0.0);
    var z = 0.0;

    for (var step = 0; step < 50; step = step + 1) {
        var p = z * normalize(vec3<f32>(I + I, 0.0) - vec3<f32>(resolution.x, resolution.y, resolution.y));

        p.z = p.z + 5.0 + cos(t) * pulseUniforms.height;

        let angle = p.y * 0.5;
        let rotMat = mat2x2<f32>(
            cos(angle + 0.0),
            cos(angle + 33.0),
            cos(angle + 11.0),
            cos(angle + 0.0),
        );

        let xz = (rotMat * vec2<f32>(p.x, p.z)) / max(p.y * 0.1 + 1.0, 0.1);
        p = vec3<f32>(xz.x, p.y, xz.y);

        var freq = 2.0;
        for (var turbLoop = 0; turbLoop < 8; turbLoop = turbLoop + 1) {
            let yzx = vec3<f32>(p.y, p.z, p.x);
            p = p + cos((yzx - vec3<f32>(t / 0.1, t, freq)) * freq * pulseUniforms.turbulence) / freq;
            freq = freq / 0.6;
        }

        let dist = 0.01 + abs(length(vec2<f32>(p.x, p.z)) + p.y * 0.3 - 0.5) / 7.0;
        z = z + dist;

        let color = (sin(z / 3.0 + pulseUniforms.baseColor * pulseUniforms.colorShift) + 1.1) / max(dist, 0.0001);
        O = O + color * pulseUniforms.intensity;
    }

    return tanhApprox(O / 1000.0);
}

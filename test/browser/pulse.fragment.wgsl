const U_SPEED: f32 = 0.1;
const U_INTENSITY: f32 = 1.0;
const U_HEIGHT: f32 = 1.0;
const U_TURBULENCE: f32 = 1.0;
const U_COLOR_SHIFT: f32 = 1.0;

fn tanhApprox(x: vec4<f32>) -> vec4<f32> {
    let x2 = x * x;
    return x * (vec4<f32>(3.0) + x2) / (vec4<f32>(3.0) + 3.0 * x2);
}

@fragment
fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let t: f32 = globalUniforms.timestamp * 0.001 * U_SPEED;
    let resolution = vec2<f32>(400.0, 400.0);
    let I = vec2<f32>(fragCoord.x, resolution.y - fragCoord.y);

    var O = vec4<f32>(0.0);
    var z = 0.0;

    for (var step = 0; step < 50; step = step + 1) {
        var p = z * normalize(vec3<f32>(I + I, 0.0) - vec3<f32>(resolution.x, resolution.y, resolution.y));

        p.z = p.z + 5.0 + cos(t) * U_HEIGHT;

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
            p = p + cos((yzx - vec3<f32>(t / 0.1, t, freq)) * freq * U_TURBULENCE) / freq;
            freq = freq / 0.6;
        }

        let dist = 0.01 + abs(length(vec2<f32>(p.x, p.z)) + p.y * 0.3 - 0.5) / 7.0;
        z = z + dist;

        let color = (sin(z / 3.0 + vec4<f32>(7.0, 2.0, 3.0, 0.0) * U_COLOR_SHIFT) + 1.1) / max(dist, 0.0001);
        O = O + color * U_INTENSITY;
    }

    return tanhApprox(O / 1000.0);
}

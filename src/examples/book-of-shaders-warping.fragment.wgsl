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

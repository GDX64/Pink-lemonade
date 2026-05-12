struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0)
var noiseTex: texture_2d<f32>;

fn heatmap(tRaw: f32) -> vec3f {
    let t = clamp(tRaw, 0.0, 1.0);
    if t < 0.33 {
        return mix(vec3f(0.02, 0.02, 0.08), vec3f(0.0, 0.65, 1.0), t / 0.33);
    }
    if t < 0.66 {
        return mix(vec3f(0.0, 0.65, 1.0), vec3f(1.0, 0.9, 0.0), (t - 0.33) / 0.33);
    }
    return mix(vec3f(1.0, 0.9, 0.0), vec3f(1.0, 0.1, 0.02), (t - 0.66) / 0.34);
}

// fn cubicWeights(t: f32) -> array<f32, 4> {
//   let t2 = t * t;
//   let t3 = t2 * t;
//   let w0 = -0.5 * t3 + t2 - 0.5 * t;
//   let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
//   let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
//   let w3 = 0.5 * t3 - 0.5 * t2;
//   return array<f32, 4>(w0, w1, w2, w3);
// }

// fn sampleBicubic(tex: texture_2d<f32>, uv: vec2f) -> f32 {
//   let size = vec2i(textureDimensions(tex));
//   let coord = uv * vec2f(size) - vec2f(0.5, 0.5);
//   let base = vec2i(floor(coord));
//   let frac = fract(coord);

//   let wx = cubicWeights(frac.x);
//   let wy = cubicWeights(frac.y);

//   var accum = 0.0;
//   for (var j = 0u; j < 4u; j = j + 1u) {
//     for (var i = 0u; i < 4u; i = i + 1u) {
//       let sx = clamp(base.x + i32(i) - 1, 0, size.x - 1);
//       let sy = clamp(base.y + i32(j) - 1, 0, size.y - 1);
//       let sample = textureLoad(tex, vec2i(sx, sy), 0).x;
//       accum = accum + sample * wx[i] * wy[j];
//     }
//   }

//   return clamp(accum, 0.0, 1.0);
// }

@vertex
fn vsMain(@location(0) position: vec2f) -> VertexOut {
    var out: VertexOut;
    out.position = vec4f(position, 0.0, 1.0);
    out.uv = position * 0.5 + vec2f(0.5, 0.5);
    return out;
}

@fragment
fn fsMain(@location(0) uv: vec2f) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(noiseTex));
    let sx = clamp(i32(uv.x * f32(size.x)), 0, size.x - 1);
    let sy = clamp(i32(uv.y * f32(size.y)), 0, size.y - 1);
    let n = textureLoad(noiseTex, vec2i(sx, sy), 0).x;
    return vec4f(heatmap(n), 1.0);
}

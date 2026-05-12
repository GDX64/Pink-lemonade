struct Params {
    minX: f32,
    minY: f32,
    scaleX: f32,
    scaleY: f32,
    pointCount: u32,
    texWidth: u32,
    texHeight: u32,
    minCount: f32,
    invCountRange: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> points: array<vec2f>;

@group(0) @binding(1)
var heatOut: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> params: Params;

fn lowerBoundX(target_value: f32) -> u32 {
    var left = 0u;
    var right = params.pointCount;

    loop {
        if left >= right {
            break;
        }

        let mid = left + (right - left) / 2u;
        if points[mid].x < target_value {
            left = mid + 1u;
        } else {
            right = mid;
        }
    }

    return left;
}

@compute @workgroup_size(8, 8)
fn buildHeatmap(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= params.texWidth || gid.y >= params.texHeight {
        return;
    }

    let invScaleX = 1.0 / params.scaleX;
    let xMin = params.minX + f32(gid.x) * invScaleX;
    let xMax = xMin + invScaleX;
    let start = lowerBoundX(xMin);
    let end = select(lowerBoundX(xMax), params.pointCount, gid.x + 1u >= params.texWidth);

    var count = 0u;
    for (var i = start; i < end; i = i + 1u) {
        let p = points[i];
        let y = u32(clamp((p.y - params.minY) * params.scaleY, 0.0, f32(params.texHeight - 1u)));
        if y == gid.y {
            count = count + 1u;
        }
    }

    let intensity = clamp((f32(count) - params.minCount) * params.invCountRange, 0.0, 1.0);
    textureStore(heatOut, vec2i(gid.xy), vec4f(intensity, 0.0, 0.0, 1.0));
}

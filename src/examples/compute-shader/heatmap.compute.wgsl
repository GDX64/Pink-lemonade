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

// One slot per thread — each thread accumulates its partial count for this pixel.
var<workgroup> localCounts: array<u32, 64>;

// Dispatch as (texWidth, texHeight, 1).
// Each workgroup owns one output pixel; its 64 threads split the point scan.
@compute @workgroup_size(64)
fn buildHeatmap(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let pixelX = wid.x;
    let pixelY = wid.y;

    // Each thread scans points[lid.x, lid.x+64, lid.x+128, ...].
    var myCount = 0u;
    var i = lid.x;
    while i < params.pointCount {
        let p = points[i];
        let px = u32(clamp((p.x - params.minX) * params.scaleX, 0.0, f32(params.texWidth  - 1u)));
        let py = u32(clamp((p.y - params.minY) * params.scaleY, 0.0, f32(params.texHeight - 1u)));
        if px == pixelX && py == pixelY {
            myCount += 1u;
        }
        i += 64u;
    }

    localCounts[lid.x] = myCount;
    workgroupBarrier();

    // Thread 0 reduces and writes the final pixel value.
    if lid.x == 0u {
        var total = 0u;
        for (var j = 0u; j < 64u; j++) {
            total += localCounts[j];
        }
        let intensity = clamp((f32(total) - params.minCount) * params.invCountRange, 0.0, 1.0);
        textureStore(heatOut, vec2i(i32(pixelX), i32(pixelY)), vec4f(intensity, 0.0, 0.0, 1.0));
    }
}

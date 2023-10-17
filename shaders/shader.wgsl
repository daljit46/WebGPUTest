struct Uniforms {
    offset: vec2f,
    scale: f32,
    windowWidth: i32,
    windowHeight: i32,
    max_iterations: f32,
};

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@group(0) @binding(0) var<uniform> uUniformData: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

fn mandlebrot_iterations(c: vec2f) -> f32 {
    var z = vec2f(0.0, 0.0);
    var i: f32 = 0;
    while (i < uUniformData.max_iterations) {
        let zxx = z.x * z.x;
        let zyy = z.y * z.y;
        z = vec2f(zxx - zyy, 2.0 * z.x * z.y) + c;
        if (zxx + zyy > 4.0) {
            break;
        }
        i = i + 1.0;
    }
    return i;
}

fn hsv2rgb(c: vec3f) -> vec3f {
    let K = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);

    let v = vec3f(clamp(p.x - K.x, 0.0, 1.0),
                clamp(p.y - K.x, 0.0, 1.0),
                clamp(p.z - K.x, 0.0, 1.0));
    return c.z * mix(K.xxx, v, c.y);
}

fn location_color(c: vec2f) -> vec3f {
    // skip computation inside bulbs
    // see https://iquilezles.org/articles/mset1bulb
    // see https://iquilezles.org/articles/mset2bulb
    let c2 = dot(c, c);
    if( 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 ){
        return vec3f(0.0, 0.0, 0.0);
    }
    if( 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 ){
        return vec3f(0.0, 0.0, 0.0);
    }

    var i: f32 = mandlebrot_iterations(c);

    let iterations = i / uUniformData.max_iterations;

    var brightness = 1.0;

    if(i == uUniformData.max_iterations) {
        brightness = 0.0;
    }

    return hsv2rgb(vec3f(iterations, 1.0, brightness));
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f{
    let largest_dim = max(f32(uUniformData.windowWidth), f32(uUniformData.windowHeight));
    let scale_factor = 5 / uUniformData.scale;
    let cx = ((in.position.x - uUniformData.offset.x)/ largest_dim - 0.5) * scale_factor;
    let cy = ((in.position.y - uUniformData.offset.y)/ largest_dim - 0.35) * scale_factor;
    let color = location_color(vec2f(cx, -cy));

    return vec4f(color, 1.0);
}

@group(0) @binding(0) var<uniform> uTime: f32;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

// Max iterations
const MAX_ITERATIONS: i32 = 500;

fn mandlebrot_iterations(c: vec2f) -> i32 {
    var z = vec2f(0.0, 0.0);
    var i: i32 = 0;
    while (i < MAX_ITERATIONS) {
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (z.x * z.x + z.y * z.y > 4.0) {
            break;
        }
        i = i + 1;
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
    var i: i32 = mandlebrot_iterations(c);

    let iterations: f32 = f32(i) / f32(MAX_ITERATIONS);

    let hue = iterations;
    let saturation = 1.0;
    var brightness = 1.0;

    if(i == MAX_ITERATIONS) {
        brightness = 0.0;
    }

    return hsv2rgb(vec3f(hue, saturation, brightness));
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f{
    var c = ((in.position.xy / 600) - vec2f(0.5, 0.5))  * 3;
    var color = location_color(vec2f(c.x, -c.y));

    return vec4f(color, 1.0);
}

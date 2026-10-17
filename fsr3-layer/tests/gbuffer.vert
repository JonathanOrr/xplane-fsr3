#version 450
// fake_xplane --raster: the scene as geometry, so viewport jitter moves it like a real G-buffer pass.
// Vertices 0-5: the wall (10 m, world-fixed); 6-11: the cockpit panel (1 m, moves with the camera).
layout(push_constant) uniform Push {
    mat4 proj;
    vec4 cam;    // camera world position
    vec4 frust;  // aspect / f, 1 / f
} pc;
layout(location = 0) out vec3 v_pos;  // wall: world position; panel: view (local) position
layout(location = 1) flat out uint v_kind;

const vec2 kCorner[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));

void main()
{
    int  q = gl_VertexIndex / 6;
    vec2 c = kCorner[gl_VertexIndex % 6];
    vec3 view;
    if (q == 0)
    {
        float z = 10.0;
        view    = vec3(mix(-1.5, 1.5, c.x) * z * pc.frust.x, mix(-1.5, 1.5, c.y) * z * pc.frust.y, -z);
        v_pos   = view + pc.cam.xyz;
    }
    else
    {
        float z = 1.0;  // rows with uv.v >= 0.75, i.e. ndc.y >= 0.5
        view    = vec3(mix(-1.5, 1.5, c.x) * z * pc.frust.x, mix(0.5, 1.5, c.y) * z * pc.frust.y, -z);
        v_pos   = view;
    }
    v_kind      = uint(q);
    gl_Position = pc.proj * vec4(view, 1.0);
}

#version 450
// fake_xplane --raster G-buffer colour (EASU input). gbuffer_pos / gbuffer_vel are the depth and
// stencil views of the pass's depth/stencil buffer, as in X-Plane; the stencil reference is per draw.
// wallColor / panelColor match fake_xplane.cpp exactly (the CPU ground truth).
layout(location = 0) in vec3 v_pos;
layout(location = 1) flat in uint v_kind;
layout(location = 0) out vec4 o_color;

vec3 wallColor(vec2 w)
{
    const float a = 0.2;
    float cs = cos(a), sn = sin(a);
    float u = w.x * cs + w.y * sn, v = -w.x * sn + w.y * cs;
    bool  k = ((int(floor(u / 0.035)) + int(floor(v / 0.035))) & 1) != 0;
    return k ? vec3(0.9, 0.8, 0.3) : vec3(0.15, 0.2, 0.6);
}

vec3 panelColor(vec2 p)
{
    float dx = p.x * 2.0 - round(p.x * 2.0), dy = p.y * 4.0 - 1.6;
    float r      = sqrt(dx * dx + dy * dy);
    float ring   = 0.5 + 0.5 * cos(r * 90.0);
    float stripe = (int(floor(p.x / 0.006)) & 1) != 0 ? 0.8 : 0.3;
    return vec3(0.2 + 0.6 * ring, 0.25 + 0.5 * stripe * ring, 0.3);
}

void main()
{
    o_color = vec4(v_kind == 0u ? wallColor(v_pos.xy) : panelColor(v_pos.xy), 1.0);
}

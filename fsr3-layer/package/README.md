# FSR 3.1 for X-Plane 12 (Linux)

X-Plane 12 can render at a lower resolution and upscale with AMD FSR 1. This mod swaps that upscaler
for **AMD FSR 3.1**, which builds each frame from several previous ones. Gauge needles, text and
distant detail come out much sharper than with FSR 1, sometimes sharper than native resolution.

You keep using X-Plane's own setting. Nothing in X-Plane is modified, and removing the mod puts
everything back as it was.

## Requirements

- Linux, X-Plane 12 (Vulkan). Tested on X-Plane 12.4.4-b4.
- Any Linux distribution that runs X-Plane 12.
- A GPU that runs X-Plane 12. So far tested only on an AMD Radeon RX 9070 XT with the Mesa (RADV) driver.
- Not for VR. In VR, X-Plane's own FSR 1 is used.

## Install

1. **Put the `xplane-fsr3` folder inside your home folder or inside the `X-Plane 12` folder.**
   Steam's runtime can't see folders anywhere else.
2. **Start X-Plane through the launcher.**
   - **Steam:** right-click X-Plane 12 → Properties → General → Launch Options, and enter the full
     path to the launcher followed by `%command%`:
     ```
     "/home/you/xplane-fsr3/xplane-fsr3" %command%
     ```
     Keep the quotes. If you already use something like `gamemoderun %command%`, put the launcher
     in front: `"/home/you/xplane-fsr3/xplane-fsr3" gamemoderun %command%`.
   - **Without Steam:** start X-Plane with the launcher in front of it, for example from a desktop
     shortcut:
     ```
     "/home/you/xplane-fsr3/xplane-fsr3" "/home/you/X-Plane 12/X-Plane-x86_64"
     ```
3. **In X-Plane, open Settings → Graphics and set *Rendering Resolution (FSR Supersampling)* to any
   setting other than Off.** The setting picks the render resolution exactly as before; FSR 3.1
   does the upscaling instead of FSR 1. When it is Off, X-Plane renders at native resolution and
   the mod does nothing.

X-Plane's own sharpening pass still runs after the upscale, exactly as it did after FSR 1.

**GPU:** the mod uses whichever GPU X-Plane runs on; it changes nothing about how X-Plane picks one.
The first line of `logs/status.txt` names it. Only if X-Plane picks the wrong one (it can happen on
some laptops), put the usual Mesa variables in front of the launcher, with the card's PCI address
from `lspci` (here `03:00.0`):
```
DRI_PRIME=pci-0000_03_00_0 MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 "/home/you/xplane-fsr3/xplane-fsr3" %command%
```

## Is it working?

Open `logs/status.txt` in the mod's folder while X-Plane runs. It is updated every 10 seconds:

| Status | Meaning |
|---|---|
| `ACTIVE` | FSR 3.1 is upscaling every frame. It also shows the resolutions and the GPU time it takes. |
| `LOADED` / `WAITING` | The mod is loaded, but X-Plane's FSR is off or no flight is loaded yet. |
| `PARTLY ACTIVE` / `NOT USED` | Some or all frames used FSR 1. The line gives the reason, e.g. VR. |
| `UNSUPPORTED X-Plane version` | X-Plane changed the parts this mod relies on. FSR 1 is used; look for a newer version of the mod. |
| no `status.txt` at all | The mod did not load. Check the launch-option path and step 1 of Install. |

If the mod can't do its job for any reason, it steps aside and X-Plane's own FSR 1 runs as normal.

## What to expect

- **Image:** much sharper than FSR 1, especially gauges, text and fine detail. Zooming and camera
  movement are clean.
- **Trails:** fast-moving gauge needles and spinning propellers can leave faint trails, because
  FSR 3.1 is told how the camera moves but not how individual objects move.
- **Performance:** FSR 3.1 does more GPU work than FSR 1. On the test system (RX 9070 XT,
  5504×2166 output) it took about 2 ms per frame: 69 fps against 79 fps with FSR 1 at the same
  spot. With FSR 1 shown, the mod made no measurable difference to fps (79 fps, against 78 without it).

## Turning it off

- **For good:** remove the launcher from X-Plane's launch options (or your shortcut) and delete the
  folder.
- **To compare:** set X-Plane's FSR setting to Off for native resolution. To compare with FSR 1,
  start X-Plane without the launcher.

## Files

| File | Purpose |
|---|---|
| `xplane-fsr3` | The launcher: enables the mod for this one program and starts X-Plane. |
| `libVkLayer_xplane_fsr3.so`, `VkLayer_xplane_fsr3.json` | The mod itself, a Vulkan layer. It is active only in X-Plane, never in other programs. |
| `logs/` | `status.txt` and one technical log per launch (the newest 10 are kept). Include the newest log with any bug report. |
| `LICENSE` | This mod's licence (MIT). |
| `THIRD_PARTY_NOTICES.md` | Licence of AMD's FidelityFX SDK, which this mod uses (also MIT). |

## Advanced

These are environment variables, set in front of the launcher:

| Variable | Effect |
|---|---|
| `XP_FSR3=off` | Load the mod but leave X-Plane untouched. |
| `XP_FSR3=observe` | Run FSR 3.1 without showing it. For troubleshooting only. |
| `XP_FSR3_AB=20` | Alternate between FSR 3.1 and FSR 1 every 20 s and log the frame times of each. |
| `XP_FSR3_PROFILE=1` | Log the mod's own CPU cost. |
| `XP_FSR3_OUTPUT=/some/dir` | Write the logs and `status.txt` there instead. |

## Licence

MIT; see `LICENSE`. FSR 3.1 itself is AMD's FidelityFX SDK, also under the MIT licence
(`THIRD_PARTY_NOTICES.md`). X-Plane is a trademark of Laminar Research; this mod is not affiliated
with or endorsed by Laminar Research or AMD.

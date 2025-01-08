# Viewer and Editor

- [Viewer and Editor](#viewer-and-editor)
  - [Editor](#editor)
  - [Viewer](#viewer)
    - [View During Training](#view-during-training)
    - [View After Training](#view-after-training)

Our viewer and editor is based on [viser](https://github.com/nerfstudio-project/viser)

Selected videos rendered by our editor, <span style="font-size: 18px;"><b>Please note:</b> These videos are encoded using HEVC with 10-bit HDR colors and are best viewed on a compatible display with HDR support, e.g. recent Apple devices.</span>

https://github.com/user-attachments/assets/050f1c37-2667-4f9a-927f-5fffe1de9c9e

https://github.com/user-attachments/assets/440aa492-ed06-4519-8509-7d74c75f3275

## Editor

<b>After training</b>, you could use the following command to view and edit the scene!

```bash
python basicgs/view.py output/le3d/windowlegovary --edit
```

Like most of the editors, you could use key frames and interpolations to create your own video storyboard!

https://github.com/user-attachments/assets/9e2a9755-14d5-4788-9393-7d0c7ae95486

Just set the `KeyFrames`! All the interpolation will be done automatically! (BTW, you could set the acceleration of the camera motion in the `Interpolations`!)

## Viewer

### View During Training

If you turn the `viewer` option in the yaml option on,

```yaml
viewer:
  type: Le3dViewer
  update_freq: 1
  port: 8097
```
and you could see this in the log.

```bash
INFO:websockets.server:server listening on 0.0.0.0:8097
╭──────────────── viser ────────────────╮
│             ╷                         │
│   HTTP      │ http://localhost:8097   │
│   Websocket │ ws://localhost:8097     │
│             ╵                         │
╰───────────────────────────────────────╯
2025-01-08 06:29:35,314 INFO: Viewer initialized, port: 8097
```

You could open the http://localhost:8097 in your browser to view the scene.<br/>
Here is the screenshot of the viewer, where contains the post-processing, the camera pose, and the training losses.

![image](https://github.com/user-attachments/assets/3ed3cb83-ae10-44af-8179-a5eb3d11ab13)

And you could also render the depth, final opacity, and the histogram of the scene.

https://github.com/user-attachments/assets/572131a2-bb10-4a4f-b1cb-bf1806b85c70


### View After Training

First got your trained model ready, and then you could use the following command to view the scene!

```bash
python basicgs/view.py output/le3d/windowlegovary
```

You could change the exposure, hdr tone mapping, camera pose, defocus blur, and even curve enhancement on that!

https://github.com/user-attachments/assets/49733b45-17d2-4714-b805-88272c4f74c1

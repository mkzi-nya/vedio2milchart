<h1 align="center">vedio2milchart</h1>

<p align="center">从 Milthm 游玩录像生成可导入 milplay 的谱面包</p>

录屏 → 逐帧识别 → 连击校准 → 谱面与资源打包

## 开始使用

需要 Python 3.10 以上、Node.js 18 以上、FFmpeg（含 ffprobe）和 Tesseract OCR。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm install
```

转换一段完整游玩录像：

```sh
python convert.py recording.mp4 --output build/recording --title "歌曲名 Cloudburst"
```

转换结果在 `build/recording/milplay.zip`。压缩包包含谱面、AAC-LC 编码的 M4A、封面和背景资源；谱面由 Terser 压缩，超过 1.5 MB 时会报错。中断的任务可加 `--resume` 继续使用已完成的扫描缓存。

也可以运行 `python server.py 8765`，在浏览器打开 `http://127.0.0.1:8765/` 上传视频。macOS 可直接运行 `./start.command`。

## 检查结果

运行单元测试和基本包检查：

```sh
python -m unittest discover -s tests
python verify_project.py build/recording
```

若 Milplay 仓库与本项目位于同一父目录，还可检查实际导入、手动游玩和动画压缩：

```sh
node verify_playability.cjs build/recording
node verify_session.cjs build/recording
node verify_bundle.cjs build/recording
node verify_compact_v7.cjs build/recording
```

## 项目内容

- `convert.py` 和 `automatic.py`：命令行入口与转换流程。
- `geometry.py`、`notes_v2.py`、`hold_motion_v6.py`：识别音符、长条和判定轨迹。
- `local_reconcile.py`：用画面连击数校准判定数量和时机。
- `export_v2.py`、`bundle.py`：生成谱面并打包媒体资源。
- `tests/`：自动化测试；`index.html`、`server.py`：本机上传界面。

识别结果取决于视频清晰度、遮挡和连击读数。转换报告会保留未对齐的计数区间；这些区间需要结合画面复核，不能仅凭总连击相同就断定每个音符都正确。

## 示例

`examples/雫 Cloudburst - Milplay.zip` 是可导入 milplay 的完整验收包

来源：[Milthm Rainative 全谱面留档 - 哔哩哔哩](https://b23.tv/VdJrkA9)
